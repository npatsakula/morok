# Morok

## Task planning

- Feel free to use multiple agents to explore different strategies and solutions.
- You can find Tinygrad codebase at `submodules/tinygrad`; you can use it as source
  of inspiration. If you want to run some code you can use `uv run` since Tinygrad
  has UV infrastructure.
- Avoid assumptions that are not supported by evidence; gather information using agents.
  Don't use cheap model as an agent, it's too stupid for our codebase.
- Keep the main context clear and perform hypothesis testing using agents.
- Avoid fast/hacky solutions; make them robust and scalable.
- Make sure that you understand all type signatures and their implications (some crates may
  have updated their API since the last time you used them).
- Focus on code size and performance: minimize code; use Rust's capabilities
  and ecosystem to optimize performance.
- If you or agent reached limit of 25k tokens during file read, consider chunked file
  reading (e.g. 1000 line each chunk); don't play games with symbols searching, large
  files are source of grand knowledge.

### Debugging

If you don't know where error occured in pipeline you can walk three steps analysis:
1. Check UOp structure before optimization: use `UOp::tree()` or just extract if from pipeline
   using `RUST_LOG=morok_tensor::realize=debug` with filter `rg 'range assignment complete'`.
2. Check UOp structure after optimization: use `UOp::tree()` or just extract if from pipeline
   using `RUST_LOG=morok_tensor::realize=debug` with filter `rg 'reduction simplification complete'`.
3. Check LLVM IR using `RUST_LOG=morok_codegen::llvm::cpu::render_to_module=debug` with filter `rg 'llvm ir before verification'`.

You can compare trees and IR with Tinygrad's and spot the differences. It
can help you isolate broken patterns or codegen parts.

If you need to extract operation name from `UOp`, you can use
`UOp::op().as_ref()` (you should import IR prelude by
`use morok_ir::prelude::*;`).

## Task execution

- If you revert the initial plan, you should stop what you're doing and go back to the planning phase.
- If you catch yourself in a situation where you're going to reset changes using git, check that the file
  was committed before; otherwise you may lose your work and will be unable to re-generate it.
- Keep documentation minimal and focused on the most important aspects of the code. Your code
  should be expressive and easy to understand without extra documentation.
  
### Error handling

- For library crates, use the `snafu` crate for error handling:
  - Use `.context()` method and auto-generated structures (`*Snafu`) for error context capturing.
  - Use `context` field in `Error` enum variants for external error context capturing.
- You can use `.expect()` instead of error handling if:
  - The error is the result of a crate implementation error and the user can't do anything about it.
  - The error is not expected to occur in normal operation.
  - The error is unrecoverable and the user can't do anything about it.
  
### Dependencies

- Add dependencies to `Cargo.toml` file using `cargo add` command; if there is a new minor/major version
  since the last time you used them, check the difference in interface using an agent.
  
### Testing

- Each crate has a `test/` module with unit and property-based tests; we don't write tests in-place.
- We use the `proptest` (dep) crate for property-based testing; if we use the `proptest-derive` crate, we
  put the derive under a feature `proptest`.
- We use the `test_case` (dep) crate for unit testing if a test requires different inputs in order to reduce
  code duplication and simplify test understanding.
- We add test infrastructure to the `.tokeignore` file in order to understand the codebase size better.
  
## Task evaluation

- `cargo fmt`, `cargo clippy` and `cargo test` should pass before I can perform review.
- Don't hack/simplify tests, ensure they are comprehensive and cover all edge cases.
