//! GigaAM dynamic batched inference example.
//!
//! Usage:
//!   cargo run -p morok-model --example gigaam_infer -- audio_1.wav

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::env;
use std::time::{Duration, Instant};

use morok_dtype::DType;
use morok_ir::Op;
use morok_model::audio::MelSpectrogram;
use morok_model::gigaam::{GigaAm, GigaAmBatchedJit, SubsamplingMode};
use morok_tensor::{PrepareConfig, Tensor};

const VOCAB: &[&str] = &[
    " ", "а", "б", "в", "г", "д", "е", "ж", "з", "и", "й", "к", "л", "м", "н", "о", "п", "р", "с", "т", "у", "ф", "х",
    "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я",
];
const BLANK_ID: usize = VOCAB.len();

#[derive(Clone)]
struct KernelAgg {
    elapsed: Duration,
    count: usize,
    var_names: Vec<String>,
    global_size: Option<[usize; 3]>,
    local_size: Option<[usize; 3]>,
    code_len: usize,
}

#[derive(Clone)]
struct KernelAstSummary {
    kernel_id: u64,
    top_ops: Vec<(String, usize)>,
    ast_head: String,
    has_wmma: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let t_total = Instant::now();

    let wav_path = env::args().nth(1).ok_or("usage: gigaam_infer <audio.wav>")?;
    let profile_kernels = match env::var("MOROK_PROFILE") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "yes" || v == "on"
        }
        Err(_) => false,
    };
    let profile_ast = match env::var("MOROK_PROFILE_AST") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "yes" || v == "on"
        }
        Err(_) => false,
    };
    let amx_enabled = std::env::var("MOROK_AMX").as_deref() == Ok("1");

    let t_audio = Instant::now();
    println!("Loading audio: {wav_path}");
    let waveform = load_wav(&wav_path)?;
    let dt_audio = t_audio.elapsed();
    let duration_s = waveform.len() as f32 / 16000.0;
    println!("Samples: {} ({:.1}s)", waveform.len(), duration_s);

    let t_model = Instant::now();
    println!("\nLoading GigaAM...");
    let model = GigaAm::from_hub_with_revision("vpermilp/GigaAM-v3", "ctc")?;
    let dt_model = t_model.elapsed();
    let sample_rate = model.config.sample_rate;
    println!("Loaded: {} layers, d_model={}", model.config.n_layers, model.config.d_model);

    let mel = MelSpectrogram::new(&morok_model::audio::MelConfig {
        sample_rate,
        n_fft: model.config.n_fft,
        hop_length: model.config.hop_length,
        win_length: model.config.win_length,
        n_mels: model.config.n_mels,
        center: model.config.mel_center,
    });
    let n_mels = mel.n_mels();
    let max_batch = model.config.max_batch_size;
    let max_t_mel = model.config.max_seq_len;
    let subs_kernel_size = match model.config.subsampling_mode {
        SubsamplingMode::Conv1d => model.config.subs_kernel_size,
        SubsamplingMode::Conv2d => 3,
    };
    let max_t_sub = subs_output_length(subs_kernel_size, max_t_mel);

    let total_mel_frames = mel.num_frames(waveform.len());
    if total_mel_frames == 0 {
        println!("No frames produced from audio.");
        return Ok(());
    }

    let t_mel = Instant::now();
    println!("\nExtracting mel features: [1, {n_mels}, {total_mel_frames}]");
    let mut full_mel = Tensor::full(&[1, n_mels, total_mel_frames], 0.0f32, DType::Float32)?;
    full_mel.realize().unwrap();
    {
        let mut view = full_mel.array_view_mut::<f32>()?;
        mel.forward_into(&waveform, &mut view);
    }
    full_mel.realize().unwrap();
    let full_mel_data = full_mel.as_vec::<f32>()?;
    let dt_mel = t_mel.elapsed();

    let num_chunks = total_mel_frames.div_ceil(max_t_mel);
    println!("Chunking into {} chunks of up to {} mel frames", num_chunks, max_t_mel);

    let mut mel_batch = Tensor::full(&[max_batch, n_mels, max_t_mel], 0.0f32, DType::Float32)?;
    mel_batch.realize().unwrap();
    let lengths = Tensor::from_slice(vec![0i32; max_batch]);

    let t_prepare = Instant::now();
    let mut jit = GigaAmBatchedJit::new(model);
    println!("Preparing batched JIT plan... [{max_batch}, {n_mels}, {max_t_mel}]");
    let prepare_config = PrepareConfig::from_env();

    println!("JIT optimizer config: {:?}", prepare_config.optimizer);
    println!("AMX renderer      {}", if amx_enabled { "enabled (MOROK_AMX=1)" } else { "disabled" });
    jit.prepare_with_config(&mel_batch, &lengths, &prepare_config)?;
    let dt_prepare = t_prepare.elapsed();
    println!("Plan captured.");
    print_buffer_summary(&jit)?;
    let kernel_ast_map = if profile_ast {
        let mut map: HashMap<String, KernelAstSummary> = HashMap::new();
        for pk in jit.prepared_kernels()? {
            map.entry(pk.kernel.entry_point.clone()).or_insert_with(|| summarize_kernel_ast(pk));
        }
        map
    } else {
        HashMap::new()
    };
    if profile_kernels {
        println!("Kernel profiling enabled via MOROK_PROFILE=1");
    }
    if profile_ast {
        println!("Kernel AST summaries enabled via MOROK_PROFILE_AST=1");
    }

    let t_loop = Instant::now();
    let mut dt_pack = Duration::ZERO;
    let mut dt_exec = Duration::ZERO;
    let mut dt_decode = Duration::ZERO;
    let mut by_entry_point: HashMap<String, KernelAgg> = HashMap::new();
    let mut full_text = String::new();
    for chunk_batch_start in (0..num_chunks).step_by(max_batch) {
        let b = (num_chunks - chunk_batch_start).min(max_batch);
        let mut chunk_lengths = vec![0usize; b];

        let t_pack_batch = Instant::now();
        {
            let mut view = jit.mel_mut()?.as_array_mut::<f32>()?;
            let slice = view.as_slice_mut().expect("contiguous");
            slice.fill(0.0);

            for (bi, chunk_len) in chunk_lengths.iter_mut().enumerate() {
                let chunk_idx = chunk_batch_start + bi;
                let mel_start = chunk_idx * max_t_mel;
                let valid = (total_mel_frames - mel_start).min(max_t_mel);
                *chunk_len = valid;

                for mel_bin in 0..n_mels {
                    let src = mel_bin * total_mel_frames + mel_start;
                    let dst = ((bi * n_mels) + mel_bin) * max_t_mel;
                    slice[dst..dst + valid].copy_from_slice(&full_mel_data[src..src + valid]);
                }
            }
        }

        {
            let mut view = jit.lengths_mut()?.as_array_mut::<i32>()?;
            let slice = view.as_slice_mut().expect("contiguous");
            slice.fill(0);
            for (i, len) in chunk_lengths.iter().enumerate() {
                slice[i] = *len as i32;
            }
        }
        dt_pack += t_pack_batch.elapsed();

        let t_exec_batch = Instant::now();
        let t_exec = chunk_lengths.iter().copied().max().unwrap_or(1).max(1);
        if profile_kernels {
            let profiles = jit.execute_with_vars_profiled(&[("b", b as i64), ("t", t_exec as i64)])?;
            for p in profiles {
                let e = by_entry_point.entry(p.kernel.entry_point.clone()).or_insert_with(|| KernelAgg {
                    elapsed: Duration::ZERO,
                    count: 0,
                    var_names: p.kernel.var_names.clone(),
                    global_size: p.kernel.global_size,
                    local_size: p.kernel.local_size,
                    code_len: p.kernel.code.len(),
                });
                e.elapsed += p.elapsed;
                e.count += 1;
            }
        } else {
            jit.execute_with_vars(&[("b", b as i64), ("t", t_exec as i64)])?;
        }
        dt_exec += t_exec_batch.elapsed();

        let t_decode_batch = Instant::now();
        for (bi, mel_len) in chunk_lengths.iter().enumerate() {
            let actual_sub = subs_output_length(subs_kernel_size, *mel_len);
            let text = ctc_greedy_decode_batch_item(jit.output()?, BLANK_ID, bi, max_t_sub, actual_sub);
            if !text.is_empty() {
                full_text.push_str(&text);
            }
        }
        dt_decode += t_decode_batch.elapsed();
    }
    let dt_loop = t_loop.elapsed();

    println!("Audio duration: {:.1}s", waveform.len() as f32 / sample_rate as f32);
    println!("\n--- Timings ---");
    println!("audio load      {:>9}", fmt_duration(dt_audio));
    println!("model load      {:>9}", fmt_duration(dt_model));
    println!("mel extract     {:>9}", fmt_duration(dt_mel));
    println!("jit prepare     {:>9}", fmt_duration(dt_prepare));
    println!("batch pack      {:>9}", fmt_duration(dt_pack));
    println!("batch execute   {:>9}", fmt_duration(dt_exec));
    println!("batch decode    {:>9}", fmt_duration(dt_decode));
    println!("chunk loop      {:>9}", fmt_duration(dt_loop));
    println!("total           {:>9}", fmt_duration(t_total.elapsed()));
    if duration_s > 0.0 {
        println!("loop RTF        {:>9.2}x", dt_loop.as_secs_f32() / duration_s);
        println!("exec RTF        {:>9.2}x", dt_exec.as_secs_f32() / duration_s);
    }

    if profile_kernels {
        println!("\n--- Kernel profile (aggregated by entry point) ---");

        let mut rows: Vec<(String, KernelAgg)> = by_entry_point.into_iter().collect();
        rows.sort_by(|a, b| b.1.elapsed.cmp(&a.1.elapsed));

        let mut wmma_entries = 0usize;
        if profile_ast {
            wmma_entries =
                rows.iter().filter(|(entry, _)| kernel_ast_map.get(entry).is_some_and(|s| s.has_wmma)).count();
        }

        let kernel_total = rows.iter().fold(Duration::ZERO, |acc, row| acc + row.1.elapsed);
        let total_unique = rows.len();
        let with_t = rows.iter().filter(|(_, agg)| agg.var_names.iter().any(|n| n == "t")).count();
        let with_b = rows.iter().filter(|(_, agg)| agg.var_names.iter().any(|n| n == "b")).count();
        let with_thread = rows.iter().filter(|(_, agg)| agg.var_names.iter().any(|n| n == "thread_id")).count();

        let mut by_global_size: HashMap<String, (Duration, usize)> = HashMap::new();
        for (_, agg) in &rows {
            let key = format!("{:?}", agg.global_size);
            let e = by_global_size.entry(key).or_insert((Duration::ZERO, 0));
            e.0 += agg.elapsed;
            e.1 += agg.count;
        }
        let mut gs_rows: Vec<(String, Duration, usize)> =
            by_global_size.into_iter().map(|(k, (d, c))| (k, d, c)).collect();
        gs_rows.sort_by(|a, b| b.1.cmp(&a.1));

        let mut by_base: HashMap<String, (Duration, usize, usize)> = HashMap::new();
        for (entry, agg) in &rows {
            let base = entry_base_name(entry);
            let e = by_base.entry(base).or_insert((Duration::ZERO, 0, 0));
            e.0 += agg.elapsed;
            e.1 += agg.count;
            e.2 += 1;
        }
        let mut base_rows: Vec<(String, Duration, usize, usize)> =
            by_base.into_iter().map(|(k, (d, c, u))| (k, d, c, u)).collect();
        base_rows.sort_by(|a, b| b.1.cmp(&a.1));

        println!("kernels total   {:>9}", fmt_duration(kernel_total));
        println!("unique entries  {:>9}", total_unique);
        println!("with var t/b/thr {:>5}/{:>5}/{:>5}", with_t, with_b, with_thread);
        if profile_ast {
            println!("entries w/ WMMA {:>9}", wmma_entries);
        }
        println!("base-name groups:");
        for (base, elapsed, calls, unique) in base_rows.into_iter().take(8) {
            let pct =
                if kernel_total.is_zero() { 0.0 } else { elapsed.as_secs_f64() / kernel_total.as_secs_f64() * 100.0 };
            println!(
                "  {:>7.2}% {:>9} calls {:>5} unique {:>4} base={}",
                pct,
                fmt_duration(elapsed),
                calls,
                unique,
                base
            );
        }
        println!("global_size groups:");
        for (gs, elapsed, calls) in gs_rows.into_iter().take(5) {
            let pct =
                if kernel_total.is_zero() { 0.0 } else { elapsed.as_secs_f64() / kernel_total.as_secs_f64() * 100.0 };
            println!("  {:>7.2}% {:>9} calls {:>5} gsz={}", pct, fmt_duration(elapsed), calls, gs);
        }
        println!("top kernels:");
        for (entry, agg) in rows.into_iter().take(15) {
            let elapsed = agg.elapsed;
            let count = agg.count;
            let pct =
                if kernel_total.is_zero() { 0.0 } else { elapsed.as_secs_f64() / kernel_total.as_secs_f64() * 100.0 };
            let avg = Duration::from_secs_f64(elapsed.as_secs_f64() / count as f64);
            let has_t = agg.var_names.iter().any(|n| n == "t");
            let has_b = agg.var_names.iter().any(|n| n == "b");
            println!(
                "{:>7.2}% {:>9} avg {:>9} count {:>5}  {}",
                pct,
                fmt_duration(elapsed),
                fmt_duration(avg),
                count,
                entry
            );
            println!(
                "         vars={:?} has_t={} has_b={} gsz={:?} lsz={:?} code={}B",
                agg.var_names, has_t, has_b, agg.global_size, agg.local_size, agg.code_len
            );
            if profile_ast && let Some(summary) = kernel_ast_map.get(&entry) {
                println!(
                    "         ast_id={} has_wmma={} top_ops={}",
                    summary.kernel_id,
                    summary.has_wmma,
                    format_top_ops(&summary.top_ops)
                );
                for line in summary.ast_head.lines() {
                    println!("         {}", line);
                }
            }
        }
    }

    println!("\n--- Full transcription ---");
    println!("{}", full_text);

    Ok(())
}

fn fmt_duration(d: Duration) -> String {
    if d.as_secs() >= 1 { format!("{:.2}s", d.as_secs_f64()) } else { format!("{}ms", d.as_millis()) }
}

fn fmt_bytes(bytes: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.2} GiB", b / GB)
    } else if b >= MB {
        format!("{:.2} MiB", b / MB)
    } else if b >= KB {
        format!("{:.2} KiB", b / KB)
    } else {
        format!("{} B", bytes)
    }
}

fn print_buffer_summary(jit: &GigaAmBatchedJit) -> morok_model::jit::Result<()> {
    let buffers = jit.buffers()?;
    let total_views = buffers.len();
    let total_view_bytes: usize = buffers.iter().map(|b| b.size()).sum();

    let mut by_alloc: HashMap<morok_device::BufferId, usize> = HashMap::new();
    for b in buffers {
        by_alloc.entry(b.id()).and_modify(|s| *s = (*s).max(b.size())).or_insert(b.size());
    }

    let input_ids: HashSet<morok_device::BufferId> = jit.input_buffer_ids()?.into_iter().collect();
    let output_ids: HashSet<morok_device::BufferId> = jit.output_buffers()?.into_iter().map(|b| b.id()).collect();

    let mut input_count = 0usize;
    let mut output_count = 0usize;
    let mut interm_count = 0usize;
    let mut input_bytes = 0usize;
    let mut output_bytes = 0usize;
    let mut interm_bytes = 0usize;
    let mut interm_allocs: Vec<(morok_device::BufferId, usize)> = Vec::new();

    for (id, size) in &by_alloc {
        let is_input = input_ids.contains(id);
        let is_output = output_ids.contains(id);
        if is_input {
            input_count += 1;
            input_bytes += *size;
        }
        if is_output {
            output_count += 1;
            output_bytes += *size;
        }
        if !is_input && !is_output {
            interm_count += 1;
            interm_bytes += *size;
            interm_allocs.push((*id, *size));
        }
    }

    interm_allocs.sort_by(|a, b| b.1.cmp(&a.1));

    println!("\n--- Buffer summary ---");
    println!("buffer views     {:>8}  {}", total_views, fmt_bytes(total_view_bytes));
    println!("allocations      {:>8}  {}", by_alloc.len(), fmt_bytes(by_alloc.values().sum()));
    println!("inputs           {:>8}  {}", input_count, fmt_bytes(input_bytes));
    println!("outputs          {:>8}  {}", output_count, fmt_bytes(output_bytes));
    println!("intermediate     {:>8}  {}", interm_count, fmt_bytes(interm_bytes));
    if !interm_allocs.is_empty() {
        println!("largest intermediate allocations:");
        for (id, sz) in interm_allocs.into_iter().take(12) {
            println!("  id={:<8} {}", id.0, fmt_bytes(sz));
        }
    }

    Ok(())
}

fn summarize_kernel_ast(pk: &morok_runtime::PreparedKernel) -> KernelAstSummary {
    let ast = match pk.ast.op() {
        Op::Kernel { ast, .. } => ast.clone(),
        _ => pk.ast.clone(),
    };
    let mut counts: HashMap<String, usize> = HashMap::new();
    for n in ast.toposort() {
        *counts.entry(n.op().as_ref().to_string()).or_insert(0) += 1;
    }
    let has_wmma = counts.keys().any(|k| k.eq_ignore_ascii_case("wmma"));
    let mut top_ops: Vec<(String, usize)> = counts.into_iter().collect();
    top_ops.sort_by(|a, b| b.1.cmp(&a.1));
    top_ops.truncate(8);

    let ast_head = trim_tree_head(&ast.tree(), 14);
    KernelAstSummary { kernel_id: pk.id, top_ops, ast_head, has_wmma }
}

fn trim_tree_head(tree: &str, max_lines: usize) -> String {
    tree.lines().take(max_lines).collect::<Vec<_>>().join("\n")
}

fn format_top_ops(top_ops: &[(String, usize)]) -> String {
    top_ops.iter().map(|(op, n)| format!("{}:{}", op, n)).collect::<Vec<_>>().join(", ")
}

fn entry_base_name(entry: &str) -> String {
    let mut i = entry.len();
    let bytes = entry.as_bytes();
    while i > 0 && bytes[i - 1].is_ascii_digit() {
        i -= 1;
    }
    if i > 0 && i < entry.len() && bytes[i - 1] == b'n' { entry[..i - 1].to_string() } else { entry.to_string() }
}

fn load_wav(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            reader.samples::<i16>().map(|s| s.map(|v| v as f32 / 32768.0)).collect::<Result<_, _>>()?
        }
    };
    Ok(samples)
}

fn subs_output_length(kernel_size: usize, mel_frames: usize) -> usize {
    let pad = (kernel_size - 1) / 2;
    let mut len = mel_frames;
    for _ in 0..2 {
        len = (len + 2 * pad - kernel_size) / 2 + 1;
    }
    len
}

fn ctc_greedy_decode_batch_item(
    output_buf: &morok_device::Buffer,
    blank_id: usize,
    batch_idx: usize,
    chunk_sub_frames: usize,
    max_frames: usize,
) -> String {
    let logits = output_buf.as_array::<f32>().expect("failed to read output buffer");
    let total_vocab = blank_id + 1;
    let n_frames = chunk_sub_frames.min(max_frames);
    let batch_base = batch_idx * chunk_sub_frames * total_vocab;

    let mut prev = blank_id;
    let mut text = String::new();
    let mut nan_frames = 0usize;
    for t in 0..n_frames {
        let base = batch_base + t * total_vocab;
        let best = (0..total_vocab)
            .max_by(|&a, &b| {
                let av = logits[base + a];
                let bv = logits[base + b];
                match av.partial_cmp(&bv) {
                    Some(ord) => ord,
                    None => match (av.is_nan(), bv.is_nan()) {
                        (true, true) => Ordering::Equal,
                        (true, false) => Ordering::Less,
                        (false, true) => Ordering::Greater,
                        (false, false) => Ordering::Equal,
                    },
                }
            })
            .unwrap();
        if logits[base + best].is_nan() {
            nan_frames += 1;
        }
        if best != blank_id && best != prev {
            text.push_str(VOCAB[best]);
        }
        prev = best;
    }
    if nan_frames > 0 {
        eprintln!("warning: batch {} had {}/{} frames with NaN best logit", batch_idx, nan_frames, n_frames);
    }
    text
}
