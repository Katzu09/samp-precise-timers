# samp-precise-timers ⌚

Developed for [net4game.com](https://net4game.com) (RolePlay), this SA-MP plugin provides precise timers for the server. It is written in [Rust](https://rust-lang.org), a memory-safe language.

## Installation

Simply install to your project:

```bash
sampctl package install bmisiak/samp-precise-timers
```

Include in your code and begin using the library:

```pawn
#include <samp-precise-timers>
```

### Why rewrite timers?

SA-MP's built-in timers experience progressively worse trigger inconsistency and suffer from poor performance and crashes due to invalid handling of arrays and state invalidation. This is written in Rust, which combines high-level ergonomics with the performance of a low-level language. As a result, the code is much cleaner, easily auditable, and more performant.

Take a look at the source to see the benefits.

### Performance

Most other plugins, including the native OpenMP implementation, iterate over all the timers on every server tick. This plugin, instead, keeps track of the next timer to trigger, only checking one memory location on every tick, regardless of how many timers are scheduled.

### Notes

- Creating new timers from callbacks works fine. ✔
- Supports strings and arrays properly without memory leaks. ✔
- Deleting/resetting a repeating timer from its callback works fine ✔
- Deleting/resetting a singulat timer form its PAWN callback will gracefully fail. Set a new one instead. :)

## Compiling

Install Rust from [rustup.rs](https://rustup.rs). Afterwards, you are two commands away from being able to compile for SA-MP, which is a 32-bit application.

### Compile for Linux servers

```
rustup target add i686-unknown-linux-gnu
```

Then, enter the project directory and execute:

```
cargo build --target=i686-unknown-linux-gnu --release
```

### Compile for Windows servers

**Note:** You might need to install **Visual Studio Build Tools**.

```
rustup target add i686-pc-windows-msvc
```

Then, enter the project directory and execute:

```
cargo build --target=i686-pc-windows-msvc --release
```

## Contributing

If you like the code and would like to help out, feel free to submit a pull request. Let me know at bm+code@net4game.com if you would like to join our team. 👋
