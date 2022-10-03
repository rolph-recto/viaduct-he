// build script to add Homebrew library path for Apple Silicon Macs.
// this is necessary for linking to the libCbcSolver library
// used by the egg library
fn main() {
    println!("cargo:rustc-link-search=/opt/homebrew/lib/")
}