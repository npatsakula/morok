use svod_dtype::{AddrSpace, DType, DeviceSpec};
use svod_ir::ops;
use svod_ir::{
    AxisId, AxisType, BufferizeOpts, CallInfo, CanonicalGraph, ConstValue, KernelInfo, Op, ProgramInfo, ReduceOp,
    RendererDevice, SInt, UOp, WmmaMetadata,
};

fn fixture(name: &str) -> std::sync::Arc<UOp> {
    match name {
        "weak_int_add" => {
            let lhs = UOp::const_(DType::WeakInt, ConstValue::Int(7));
            let rhs = UOp::const_(DType::Int32, ConstValue::Int(2));
            lhs.try_add(&rhs).expect("weak integer promotion")
        }
        "weak_float_neg_zero" => UOp::const_(DType::WeakFloat, ConstValue::Float(-0.0))
            .try_add(&UOp::native_const(1.0f32))
            .expect("weak float commitment"),
        "invalid_where" => {
            let condition = UOp::native_const(true);
            let value = UOp::const_(DType::Float16, ConstValue::Float(1.0));
            UOp::try_where(condition, value, UOp::invalid_marker()).expect("Invalid branch dtype derivation")
        }
        "scalar_stack" => UOp::stack(smallvec::smallvec![UOp::native_const(1i32), UOp::native_const(2i32)]),
        "shaped_stack" => {
            let row0 = UOp::stack(smallvec::smallvec![UOp::native_const(1i32), UOp::native_const(2i32)]);
            let row1 = UOp::stack(smallvec::smallvec![UOp::native_const(3i32), UOp::native_const(4i32)]);
            UOp::stack(smallvec::smallvec![row0, row1])
        }
        "buffer" => UOp::buffer(4, 8, DType::Float32, AddrSpace::Global, Some(DeviceSpec::Cpu)),
        "scalar_load" | "gated_load" => {
            let param = UOp::param(0, 16, DType::Float32, None);
            let index = UOp::const_(DType::WeakInt, ConstValue::Int(3));
            let indexed = UOp::index().buffer(param).indices(vec![index]).call().unwrap();
            if name == "gated_load" {
                let gate = UOp::const_(DType::WeakInt, ConstValue::Int(3))
                    .try_cmplt(&UOp::const_(DType::WeakInt, ConstValue::Int(5)))
                    .expect("gated LOAD comparison");
                UOp::load().index(indexed).alt(UOp::native_const(0.0f32)).gate(gate).call()
            } else {
                UOp::load().index(indexed).call()
            }
        }
        "range_split_outer" => UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(5).child(0), AxisType::Weak),
        "range_split_inner" => UOp::range_axis(UOp::index_const(2), AxisId::Renumbered(5).child(1), AxisType::Weak),
        "range_split_nested" => {
            UOp::range_axis(UOp::index_const(3), AxisId::Renumbered(5).child(1).child(0), AxisType::Weak)
        }
        "scalar_store" => {
            let output = UOp::param(0, 8, DType::Float32, None);
            UOp::index()
                .buffer(output)
                .indices(vec![UOp::index_const(2)])
                .call()
                .unwrap()
                .store(UOp::native_const(3.0f32))
        }
        "mixed_valid_load" => {
            let input = UOp::param(0, 8, DType::Float32, None);
            let index = UOp::index_const(3);
            let valid = index.try_cmplt(&UOp::index_const(4)).expect("index validity comparison");
            let plain = UOp::load()
                .index(UOp::index().buffer(input.clone()).indices(vec![UOp::index_const(2)]).call().unwrap())
                .call();
            let gated =
                UOp::load().index(UOp::index().buffer(input).indices(vec![index.valid(valid)]).call().unwrap()).call();
            UOp::stack(smallvec::smallvec![plain, gated])
        }
        "copy" => UOp::param(0, 8, DType::Float32, Some(DeviceSpec::Cpu)).copy(DeviceSpec::Cuda { device_id: 0 }),
        "allreduce" => UOp::allreduce(UOp::param(0, 8, DType::Float32, None), DeviceSpec::Cpu, ReduceOp::Add),
        "multi_output_call" => {
            let body = UOp::tuple(smallvec::smallvec![UOp::native_const(1i32), UOp::native_const(2i32)]);
            body.call(smallvec::smallvec![], CallInfo { name: Some("pair".into()), ..Default::default() })
        }
        "padded_reduction" => {
            let source = UOp::stack(smallvec::smallvec![
                UOp::native_const(1.0f32),
                UOp::native_const(2.0f32),
                UOp::native_const(3.0f32),
            ]);
            source
                .try_pad(&[(SInt::from(1usize), SInt::from(2usize))])
                .unwrap()
                .try_reduce_axis(ReduceOp::Add, vec![0])
                .unwrap()
        }
        "local_wmma_staging" => {
            let a = UOp::stage(
                UOp::stack(smallvec::smallvec![UOp::const_(DType::Float16, ConstValue::Float(1.0))]),
                vec![],
                BufferizeOpts::local(),
            );
            let b = UOp::stage(
                UOp::stack(smallvec::smallvec![UOp::const_(DType::Float16, ConstValue::Float(2.0))]),
                vec![],
                BufferizeOpts::local(),
            );
            UOp::wmma(
                a,
                b,
                UOp::stack(smallvec::smallvec![UOp::native_const(0.0f32)]),
                WmmaMetadata {
                    name: "common_wmma".into(),
                    dims: (16, 8, 16),
                    dtype_in: DType::Float16,
                    dtype_out: DType::Float32,
                    device: RendererDevice::Cpu,
                    threads: 1,
                    upcast_axes: None,
                    reduce_axes: vec![],
                },
            )
        }
        "symbolic_function" => {
            let formal_dim = UOp::scalar_param(1, Some("n".into()), DType::Int32, 1, 8);
            let formal_extent = formal_dim
                .try_mul(&formal_dim.const_like(2i32))
                .unwrap()
                .try_add(&formal_dim.const_like(1i32))
                .unwrap();
            let formal =
                UOp::param_with_shape(0, &smallvec::smallvec![SInt::Symbolic(formal_extent)], DType::Float32, None);
            let body = UOp::tuple(smallvec::smallvec![formal.clone(), formal.try_sqrt().unwrap()]);

            let actual_dim = UOp::variable("m".into(), 1, 8, DType::Int32);
            let actual_extent = actual_dim
                .try_mul(&actual_dim.const_like(2i32))
                .unwrap()
                .try_add(&actual_dim.const_like(1i32))
                .unwrap();
            let actual =
                UOp::param_with_shape(7, &smallvec::smallvec![SInt::Symbolic(actual_extent)], DType::Float32, None);
            body.try_function(
                smallvec::smallvec![actual, actual_dim],
                CallInfo { name: Some("symbolic".into()), ..Default::default() },
            )
            .unwrap()
            .try_gettuple(1)
            .unwrap()
        }
        "program_info" => {
            let input = UOp::param(0, 16, DType::Float32, None);
            let output = UOp::param(2, 16, DType::Float32, None);
            let index = UOp::index_const(0);
            let load =
                UOp::load().index(UOp::index().buffer(input).indices(vec![index.clone()]).call().unwrap()).call();
            let store = UOp::index().buffer(output).indices(vec![index]).call().unwrap().store(load);
            let variable = UOp::variable("n".into(), 1, 16, DType::Int32);
            let Op::Param(ops::Param { shape, arg }) = variable.op() else { unreachable!() };
            let mut arg = arg.clone();
            arg.slot = 1;
            let variable = UOp::new(Op::Param(ops::Param { shape: shape.clone(), arg }), DType::Int32);
            let sink = UOp::sink_with_info(
                vec![store, variable.clone()],
                KernelInfo { name: Some("non_default".into()), ..Default::default() },
            );
            let mut info = ProgramInfo::from_sink(&sink, DeviceSpec::Cuda { device_id: 1 });
            info.global_size = [variable.clone(), UOp::index_const(2), UOp::index_const(1)];
            info.local_size = Some([UOp::index_const(4), UOp::index_const(1), UOp::index_const(1)]);
            UOp::program(sink, info, None, None, None)
        }
        _ => panic!("unknown fixture {name:?}"),
    }
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let name = args.first().expect("usage: canonical_fixture <fixture> [--verbose]");
    let root = fixture(name);
    let stage = if name == "program_info" { "program" } else { "tensor" };
    let graph = if args.iter().any(|arg| arg == "--verbose") {
        CanonicalGraph::from_root_verbose(stage, &root)
    } else {
        CanonicalGraph::from_root(stage, &root)
    }
    .expect("fixture must have a valid shape");
    println!("{}", graph.to_pretty_json().expect("canonical graph must serialize"));
}
