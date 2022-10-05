/// typechecker.rs
/// checks for dimensionality constraints on source expressions

use crate::lang::{*, SourceExpr::*};

pub struct TypeChecker;

impl TypeChecker {
    pub fn new() -> Self {
        TypeChecker {}
    }

    pub fn run(&self, program: &SourceProgram) -> Result<usize, String> {
        let mut store: im::HashMap<&str, usize> =
            im::HashMap::from_iter(
                program.inputs.iter().map(|input|
                    (input.0.as_ref(), input.1.len())
                )
            );
        self.runWithStores(&program.expr, &store)
    }

    fn runWithStores(&self, expr: &SourceExpr, store: &im::HashMap<&str, usize>) -> Result<usize, String> {
        match expr {
            ForNode(index, extent, body) => {
                let dim = self.runWithStores(body, store)?;
                Ok(dim+1)
            }

            ReduceNode(_, body) => {
                let body_dim = self.runWithStores(body, store)?;
                if body_dim > 0 {
                    Ok(body_dim-1)

                } else {
                    Err(String::from("cannot reduce scalar value"))
                }
            },

            OpNode(_, expr1, expr2) => {
                let dim1 = self.runWithStores(expr1, store)?;
                let dim2 = self.runWithStores(expr2, store)?;

                if dim1 == dim2 {
                    Ok(dim1)
                } else {
                    Err(String::from("operands must have the same dimension"))
                }
            }

            IndexingNode(arr, index_list) => {
                let arr_dim: usize =
                    *store.get(arr.as_str())
                         .ok_or(format!("array {} not in store", arr))?;
                let num_indices = index_list.len();
                if arr_dim >= num_indices {
                    Ok(arr_dim - num_indices)

                } else {
                    Err(format!("array with {} dimensions cannot have index list of length {}", arr_dim, num_indices))
                }
            },

            LiteralNode(_) => Ok(0)
        }
    }
}