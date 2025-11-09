# Morok

## Task planning

- Feel free to use multiple agents to explore different strategies and solutions.
- Avoid assumptions that are not supported by evidence; gather information using agents.
- Keep the main context clear and perform hypothesis testing using agents.
- Avoid fast/hacky solutions; make them robust and scalable.
- Make sure that you understand all type signatures and their implications (some crates may
  have updated their API since the last time you used them).
- Focus on code size and performance: minimize code; use Rust's capabilities
  and ecosystem to optimize performance.

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
  
## Task evaluation

- `cargo fmt`, `cargo clippy` and `cargo test` should pass before I can perform review.
- Don't hack/simplify tests, ensure they are comprehensive and cover all edge cases.
