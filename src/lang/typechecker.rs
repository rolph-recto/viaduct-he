/// typechecker.rs
/// checks for dimensionality constraints on source expressions

use crate::lang::{*, SourceExpr::*};

pub struct TypeChecker;

impl TypeChecker {
    pub fn new() -> Self {
        TypeChecker {}
    }

    pub fn run(&self, expr: &SourceExpr, store: &ArrayEnvironment) -> Result<Shape, String> {
        self._run(expr, store, & im::HashMap::new())
    }

    fn _run(&self, expr: &SourceExpr, store: &ArrayEnvironment, index_store: &IndexEnvironment) -> Result<Shape, String> {
        match expr {
            ForNode(index, extent, body) => {
                self._run(
                    body,
                    store,
                    &index_store.update(index.clone(), *extent)
                )
            }

            ReduceNode(_, body) => {
                let mut body_shape = self._run(body, store, index_store)?;
                if body_shape.len() > 0 {
                    Ok(body_shape.split_off(1))

                } else {
                    Err(String::from("cannot reduce scalar value"))
                }
            },

            OpNode(_, expr1, expr2) => {
                let shape1 = self._run(expr1, store, index_store)?;
                let shape2 = self._run(expr2, store, index_store)?;

                if shape1.len() == shape2.len() {
                    Ok(shape1)
                } else {
                    Err(String::from("operands must have the same dimension"))
                }
            }

            IndexingNode(arr, index_list) => {
                let shape: &Shape =
                    store.get(arr)
                         .ok_or(format!("array {} not in store", arr))?;
                let sn = shape.len();
                let iln = index_list.len();
                if sn >= iln {
                    Ok(shape.clone().split_off(iln))

                } else {
                    Err(format!("array with {} dimensions cannot have index list of length {}", sn, iln))
                }
            },

            LiteralNode(_) => Ok(im::Vector::new())
        }
    }
}