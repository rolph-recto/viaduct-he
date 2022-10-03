/// expr.rs
/// Expression representation of HE programs.
/// The egg library manipulates this representation.

use egg::*;

define_language! {
    /// The language used by egg e-graph engine.
    pub enum HE {
        Num(i32),
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        "rot" = Rot([Id; 2]),
        Symbol(Symbol),
    }
}