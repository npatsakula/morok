//! GigaAM dynamic batched inference example.
//!
//! Usage:
//!   cargo run -p morok-model --example gigaam_infer -- audio_1.wav

use std::collections::{HashMap, HashSet};
use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};

use morok_arch::ctc::CtcDecoder;
use morok_dtype::DType;
use morok_ir::{ConstValue, Op, UOp};
use morok_model::audio::MelSpectrogram;
use morok_model::gigaam::{GigaAm, GigaAmBatchedJit, SubsamplingMode};
use morok_tensor::{PrepareConfig, Tensor};

#[derive(Clone)]
struct KernelAgg {
    elapsed: Duration,
    count: usize,
    var_names: Vec<String>,
    global_size: String,
    local_size: String,
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
    let debug_logits = match env::var("MOROK_DEBUG_LOGITS") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "yes" || v == "on"
        }
        Err(_) => false,
    };
    let amx_enabled = std::env::var("MOROK_AMX").as_deref() == Ok("1");
    let beam_decode_enabled = match env::var("MOROK_BEAM_DECODE") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "yes" || v == "on"
        }
        Err(_) => false,
    };

    let t_audio = Instant::now();
    println!("Loading audio: {wav_path}");
    let waveform = load_wav(&wav_path)?;
    let dt_audio = t_audio.elapsed();
    let duration_s = waveform.len() as f32 / 16000.0;
    println!("Samples: {} ({:.1}s)", waveform.len(), duration_s);

    let t_model = Instant::now();
    println!("\nLoading GigaAM...");
    let mut model = GigaAm::from_hub_with_revision("vpermilp/GigaAM-v3", "ctc")?;
    let dt_model = t_model.elapsed();
    let sample_rate = model.config.sample_rate;
    let mut decoder = if beam_decode_enabled {
        // Promote a greedy config to a beam decoder using its vocabulary.
        match &model.config.decoder {
            CtcDecoder::Greedy(g) => CtcDecoder::Beam(Box::new(morok_arch::ctc::BeamDecoder::new(
                g.vocabulary().to_vec(),
                morok_arch::ctc::BeamOpts::default(),
            ))),
            other => other.clone(),
        }
    } else {
        model.config.decoder.clone()
    };
    let blank_id = decoder.blank_id();
    let total_vocab = decoder.total_vocab();
    println!(
        "Loaded: {} layers, d_model={}, vocab_size={}, decoder={}",
        model.config.n_layers,
        model.config.d_model,
        model.config.vocab_size,
        match &decoder {
            CtcDecoder::Greedy(_) => "greedy",
            CtcDecoder::Beam(_) => "beam",
        }
    );

    let mel = MelSpectrogram::new(&morok_model::audio::MelConfig {
        sample_rate,
        n_fft: model.config.n_fft,
        hop_length: model.config.hop_length,
        win_length: model.config.win_length,
        n_mels: model.config.n_mels,
        center: model.config.mel_center,
    });
    let n_mels = mel.n_mels();
    let max_t_mel = model.config.max_mel_frames;
    let hop_length = model.config.hop_length;
    let subsampling_factor = model.config.subsampling_factor;
    let subs_kernel_size = match model.config.subsampling_mode {
        SubsamplingMode::Conv1d => model.config.subs_kernel_size,
        SubsamplingMode::Conv2d => 3,
    };

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

    // VAD-aware chunking: run Silero V5 → arch::vad::chunks_from_probs to get
    // speech-bearing sample ranges, then convert each to a (mel_start, mel_len)
    // slice into the precomputed full_mel. Pure-silence regions are dropped.
    let t_vad = Instant::now();
    println!("\nLoading Silero VAD...");
    let vad_model = morok_model::vad::SileroVad::from_hub()?;
    let mut vad = morok_model::vad::VadInference::new(vad_model)?;
    let probs = vad.probs(&waveform)?;
    // Each chunk's mel-frame count must fit max_t_mel. The chunker rounds
    // chunk boundaries to align_to-sample multiples (= subsampling_factor mel
    // frames), and the start/end snap can each shift by up to one alignment
    // step — so a chunk's mel length can grow by 2 × subsampling_factor
    // beyond the chunker's nominal max_duration. Reserve that much headroom.
    let mel_headroom = 2 * subsampling_factor;
    let encoder_capacity_secs =
        (max_t_mel.saturating_sub(mel_headroom) as f32 * hop_length as f32) / sample_rate as f32;
    let default_opts = morok_arch::vad::ChunkerOpts::default();
    let chunker_opts = morok_arch::vad::ChunkerOpts {
        sample_rate: sample_rate as u32,
        samples_per_prob: morok_model::vad::NUM_SAMPLES,
        max_duration: default_opts.max_duration.min(encoder_capacity_secs),
        strict_limit_duration: default_opts.strict_limit_duration.min(encoder_capacity_secs),
        align_to: hop_length * subsampling_factor,
        ..default_opts
    };
    let vad_chunks = morok_arch::vad::chunks_from_probs(&probs, &chunker_opts)?;
    let dt_vad = t_vad.elapsed();
    println!(
        "VAD: {} probs → {} chunks (max_chunk={:.1}s, strict={:.1}s, encoder_cap={:.1}s, align_to={} samples) in {}",
        probs.len(),
        vad_chunks.len(),
        chunker_opts.max_duration,
        chunker_opts.strict_limit_duration,
        encoder_capacity_secs,
        chunker_opts.align_to,
        fmt_duration(dt_vad),
    );

    // (mel_start, mel_len, start_sec) per chunk.
    let chunks_meta: Vec<(usize, usize, f32)> = vad_chunks
        .iter()
        .filter_map(|c| {
            let mel_start = c.start_sample / hop_length;
            let mel_end = (c.end_sample / hop_length).min(total_mel_frames);
            if mel_end <= mel_start {
                return None;
            }
            let start_sec = c.start_sample as f32 / sample_rate as f32;
            Some((mel_start, mel_end - mel_start, start_sec))
        })
        .collect();

    if chunks_meta.is_empty() {
        println!("\nNo speech detected; transcript is empty.");
        println!("Audio duration: {:.1}s", duration_s);
        return Ok(());
    }

    let num_chunks = chunks_meta.len();
    let max_batch = model.config.max_batch_size.min(num_chunks);
    model.config.max_batch_size = max_batch;
    println!(
        "Chunking into {} VAD chunks of up to {} mel frames; JIT batch bound {}",
        num_chunks, max_t_mel, max_batch
    );

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
    if std::env::var("MOROK_DUMP_KERNELS").as_deref() == Ok("1") {
        for kernel in jit.prepared_kernels().unwrap() {
            println!("{}", kernel.kernel.code);
        }
    }
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
    let mut chunk_texts: Vec<String> = Vec::with_capacity(num_chunks);
    for chunk_batch_start in (0..num_chunks).step_by(max_batch) {
        let b = (num_chunks - chunk_batch_start).min(max_batch);
        let mut chunk_lengths = vec![0usize; b];

        let t_pack_batch = Instant::now();
        {
            let mut view = jit.mel_mut()?.as_array_mut::<f32>()?;
            let slice = view.as_slice_mut().expect("contiguous");
            slice.fill(0.0);

            for (bi, chunk_len) in chunk_lengths.iter_mut().enumerate() {
                let &(mel_start, valid, _start_sec) = &chunks_meta[chunk_batch_start + bi];
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
        let t_exec_sub = subs_output_length(subs_kernel_size, t_exec);
        if profile_kernels {
            let profiles = jit.execute_with_vars_profiled(&[("b", b as i64), ("t", t_exec as i64)])?;
            for p in profiles {
                let e = by_entry_point.entry(p.kernel.entry_point.clone()).or_insert_with(|| KernelAgg {
                    elapsed: Duration::ZERO,
                    count: 0,
                    var_names: p.kernel.var_names.clone(),
                    global_size: format_launch_size(&p.kernel.global_size),
                    local_size: format_launch_local_size(p.kernel.local_size.as_ref()),
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
        let logits_array = jit.output()?.as_array::<f32>().expect("failed to read output logits");
        let logits_slice = logits_array.as_slice().expect("contiguous output logits");
        let item_stride = t_exec_sub * total_vocab;
        for (bi, mel_len) in chunk_lengths.iter().enumerate() {
            let actual_sub = subs_output_length(subs_kernel_size, *mel_len);
            let item_base = bi * item_stride;
            let item_slice = &logits_slice[item_base..item_base + item_stride];

            if debug_logits && chunk_batch_start == 0 && bi == 0 {
                let sample_n = total_vocab * actual_sub.min(4);
                let sample = &item_slice[..sample_n.min(item_slice.len())];
                let (mut min_v, mut max_v, mut nonzero) = (f32::INFINITY, f32::NEG_INFINITY, 0usize);
                for &v in sample {
                    min_v = min_v.min(v);
                    max_v = max_v.max(v);
                    if v != 0.0 {
                        nonzero += 1;
                    }
                }

                // Use the decoder's argmax helper. argmax_per_frame is on
                // GreedyDecoder; build a lightweight one keyed off the live
                // vocabulary so debug output matches the active decoder.
                let debug_decoder = morok_arch::ctc::GreedyDecoder::new(decoder.vocabulary().to_vec());
                let head_n = actual_sub.min(16);
                let first_ids = debug_decoder.argmax_per_frame(item_slice, t_exec_sub, head_n);
                let first_labels: Vec<String> =
                    first_ids
                        .iter()
                        .map(|&id| {
                            if id == blank_id { "<blank>".to_string() } else { debug_decoder.vocabulary()[id].clone() }
                        })
                        .collect();

                let all_ids = debug_decoder.argmax_per_frame(item_slice, t_exec_sub, actual_sub);
                let mut id_hist = vec![0usize; total_vocab];
                for id in all_ids {
                    id_hist[id] += 1;
                }
                let mut top_hist: Vec<(usize, usize)> =
                    id_hist.into_iter().enumerate().filter(|(_, n)| *n > 0).collect();
                top_hist.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
                println!(
                    "debug logits: sample={} min={:.6} max={:.6} nonzero={} stride={} valid={}",
                    sample.len(),
                    min_v,
                    max_v,
                    nonzero,
                    t_exec_sub,
                    actual_sub
                );
                println!("debug logits: first ids={:?}", first_ids);
                println!("debug logits: first labels={:?}", first_labels);
                println!("debug logits: top ids={:?}", top_hist.into_iter().take(6).collect::<Vec<_>>());
            }

            let text = decoder.decode(item_slice, t_exec_sub, actual_sub)?;
            let &(_, _, start_sec) = &chunks_meta[chunk_batch_start + bi];
            if !text.is_empty() {
                println!("  [{:>6.1}s] {}", start_sec, text);
            }
            chunk_texts.push(text);
        }
        dt_decode += t_decode_batch.elapsed();
    }
    let dt_loop = t_loop.elapsed();
    let full_text = chunk_texts.iter().filter(|s| !s.is_empty()).cloned().collect::<Vec<_>>().join(" ");

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
        rows.sort_by_key(|row| std::cmp::Reverse(row.1.elapsed));

        let mut wmma_entries = 0usize;
        if profile_ast {
            wmma_entries =
                rows.iter().filter(|(entry, _)| kernel_ast_map.get(entry).is_some_and(|s| s.has_wmma)).count();
        }

        let kernel_total = rows.iter().fold(Duration::ZERO, |acc, row| acc + row.1.elapsed);
        let total_unique = rows.len();
        let with_t = rows.iter().filter(|(_, agg)| agg.var_names.iter().any(|n| n == "t")).count();
        let with_b = rows.iter().filter(|(_, agg)| agg.var_names.iter().any(|n| n == "b")).count();
        let with_thread = rows.iter().filter(|(_, agg)| agg.var_names.iter().any(|n| n == "core_id")).count();

        let mut by_global_size: HashMap<String, (Duration, usize)> = HashMap::new();
        for (_, agg) in &rows {
            let key = agg.global_size.clone();
            let e = by_global_size.entry(key).or_insert((Duration::ZERO, 0));
            e.0 += agg.elapsed;
            e.1 += agg.count;
        }
        let mut gs_rows: Vec<(String, Duration, usize)> =
            by_global_size.into_iter().map(|(k, (d, c))| (k, d, c)).collect();
        gs_rows.sort_by_key(|row| std::cmp::Reverse(row.1));

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
        base_rows.sort_by_key(|row| std::cmp::Reverse(row.1));

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
                "         vars={:?} has_t={} has_b={} gsz={} lsz={} code={}B",
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

fn format_const_value(value: ConstValue) -> String {
    match value {
        ConstValue::Int(v) => v.to_string(),
        ConstValue::UInt(v) => v.to_string(),
        ConstValue::Bool(v) => v.to_string(),
        ConstValue::Float(v) => v.to_string(),
    }
}

fn format_launch_dim(dim: &Arc<UOp>) -> String {
    match dim.op() {
        Op::Const(value) => format_const_value(value.0),
        Op::DefineVar { name, .. } => name.clone(),
        Op::Bind { var, value } => format!("{}={}", format_launch_dim(var), format_launch_dim(value)),
        Op::Binary(op, lhs, rhs) => format!("({} {:?} {})", format_launch_dim(lhs), op, format_launch_dim(rhs)),
        Op::Cast { src, .. } | Op::BitCast { src, .. } | Op::After { passthrough: src, .. } => format_launch_dim(src),
        _ => dim.op().as_ref().to_string(),
    }
}

fn format_launch_size(size: &[Arc<UOp>; 3]) -> String {
    format!("[{}, {}, {}]", format_launch_dim(&size[0]), format_launch_dim(&size[1]), format_launch_dim(&size[2]))
}

fn format_launch_local_size(size: Option<&[Arc<UOp>; 3]>) -> String {
    size.map(format_launch_size).unwrap_or_else(|| "None".to_string())
}

fn print_buffer_summary(jit: &GigaAmBatchedJit) -> morok_model::jit::Result<()> {
    let buffers = jit.buffers()?;
    let total_views = buffers.len();
    let total_view_bytes: usize = buffers.iter().map(|b| b.size()).sum();

    // Dedup by `storage_id()` (per-allocation identity), not `id()` (per-handle).
    // Under arena mode, hundreds of views share one underlying allocation; keying
    // by the handle id would count each view as its own allocation and inflate
    // the totals. `total_size` here is the underlying allocation size — fixed per
    // storage and shared by every view of it.
    let mut by_alloc: HashMap<morok_device::BufferId, usize> = HashMap::new();
    for b in buffers {
        by_alloc.entry(b.storage_id()).or_insert_with(|| b.total_size());
    }

    // Resolve input handle ids → storage ids by looking each input handle up in
    // the plan's buffer table.
    let input_handle_ids: HashSet<morok_device::BufferId> = jit.input_buffer_ids()?.into_iter().collect();
    let input_storage_ids: HashSet<morok_device::BufferId> =
        buffers.iter().filter(|b| input_handle_ids.contains(&b.id())).map(|b| b.storage_id()).collect();
    let output_storage_ids: HashSet<morok_device::BufferId> =
        jit.output_buffers()?.into_iter().map(|b| b.storage_id()).collect();

    let mut input_count = 0usize;
    let mut output_count = 0usize;
    let mut interm_count = 0usize;
    let mut input_bytes = 0usize;
    let mut output_bytes = 0usize;
    let mut interm_bytes = 0usize;
    let mut interm_allocs: Vec<(morok_device::BufferId, usize)> = Vec::new();

    for (id, size) in &by_alloc {
        let is_input = input_storage_ids.contains(id);
        let is_output = output_storage_ids.contains(id);
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

    interm_allocs.sort_by_key(|row| std::cmp::Reverse(row.1));

    println!("\n--- Buffer summary ---");
    println!("buffer views     {:>8}  {}", total_views, fmt_bytes(total_view_bytes));
    println!("allocations      {:>8}  {}", by_alloc.len(), fmt_bytes(by_alloc.values().sum()));
    println!("inputs           {:>8}  {}", input_count, fmt_bytes(input_bytes));
    println!("outputs          {:>8}  {}", output_count, fmt_bytes(output_bytes));
    println!("intermediate     {:>8}  {}", interm_count, fmt_bytes(interm_bytes));
    if !interm_allocs.is_empty() {
        println!("largest intermediate allocations:");
        for (id, sz) in interm_allocs.into_iter().take(12) {
            println!("  storage_id={:<8} {}", id.0, fmt_bytes(sz));
        }
    }

    Ok(())
}

fn summarize_kernel_ast(pk: &morok_runtime::PreparedKernel) -> KernelAstSummary {
    let ast = match pk.ast.op() {
        Op::Call { body, .. } => body.clone(),
        _ => pk.ast.clone(),
    };
    let mut counts: HashMap<String, usize> = HashMap::new();
    for n in ast.toposort() {
        *counts.entry(n.op().as_ref().to_string()).or_insert(0) += 1;
    }
    let has_wmma = counts.keys().any(|k| k.eq_ignore_ascii_case("wmma"));
    let mut top_ops: Vec<(String, usize)> = counts.into_iter().collect();
    top_ops.sort_by_key(|row| std::cmp::Reverse(row.1));
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
