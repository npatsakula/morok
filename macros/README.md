# svod-macros

Procedural macros for the Svod ML compiler.

## `patterns!`

Generates a pattern matcher from declarative rewrite rules. Used by `svod-schedule`
to build the optimization engine.

```rust
use svod_schedule::patterns;

let matcher = patterns! {
    Add[x, @zero] => x,
    Mul[x, @one] => x,
    Neg(Neg(x)) => x,
};
```

See [`svod-schedule` README](../schedule/README.md) for the full DSL reference.
