/// typechecker.rs
/// checks for dimensionality constraints on source expressions

use crate::lang::{*, source::{*, SourceExpr::*}};

pub struct TypeChecker;

impl TypeChecker {
    pub fn new() -> Self {
        TypeChecker {}
    }

    pub fn run(&self, program: &SourceProgram) -> Result<usize, String> {
        let mut store: im::HashMap<&str, usize> = im::HashMap::new();
        program.input_map.iter().try_for_each(|(array, shape)|
            match store.insert(array, shape.len()) {
                Some(_) => Err(format!("duplicate bindings for {}", array)),
                None => Ok(())
            }
        )?;
        
        program.expr_map.iter().try_for_each(|(array, expr)| {
            let rhs_dims = self.run_with_stores(expr, &store)?;
            match store.insert(array, rhs_dims) {
                Some(_) => Err(format!("duplicate bindings for {}", array)),
                None => Ok(())
            }
        })?;

        Ok(store[OUTPUT_EXPR_NAME])
    }

    fn run_with_stores(&self, expr: &SourceExpr, store: &im::HashMap<&str, usize>) -> Result<usize, String> {
        match expr {
            For(index, extent, body) => {
                let dim = self.run_with_stores(body, store)?;
                Ok(dim+1)
            }

            Reduce(_, body) => {
                let body_dim = self.run_with_stores(body, store)?;
                if body_dim > 0 {
                    Ok(body_dim-1)

                } else {
                    Err(String::from("cannot reduce scalar value"))
                }
            },

            ExprOp(_, expr1, expr2) => {
                let dim1 = self.run_with_stores(expr1, store)?;
                let dim2 = self.run_with_stores(expr2, store)?;

                if dim1 == dim2 {
                    Ok(dim1)
                } else {
                    Err(String::from("operands must have the same dimension"))
                }
            }

            Indexing(arr, index_list) => {
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

            Literal(_) => Ok(0)
        }
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        TypeChecker::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::lang::parser::ProgramParser;
    use super::*;

    #[test]
    fn test_typechecker_positive() {
        let parser = ProgramParser::new();
        let typechecker = TypeChecker::new();

        let prog1 = parser.parse("42").unwrap();
        let prog2 = parser.parse("42 + 56").unwrap();
        let prog3 = parser.parse("
            input img: [16,16] from client
            for x: 16 { img[x] }
        ").unwrap();

        assert!(typechecker.run(&prog1).is_ok());
        assert!(typechecker.run(&prog2).is_ok());
        assert!(typechecker.run(&prog3).is_ok());
    }

    #[test]
    fn test_typechecker_negative() {
        let parser = ProgramParser::new();
        let typechecker = TypeChecker::new();

        let prog1 = parser.parse("sum(42)").unwrap();
        let prog2 = parser.parse("
            input img: [16,16] from client
            for x: 16 {
                for y: 16 {
                    for z: 16 { img[x][y][z] }
                }
            }
        ").unwrap();
        let prog3 = parser.parse("
            input img: [16,16] from client
            let next = img + img in
            for x: 16 {
                for y: 16 {
                    img[x][y] + next
                }
            }
        ").unwrap();

        assert!(typechecker.run(&prog1).is_err());
        assert!(typechecker.run(&prog2).is_err());
        assert!(typechecker.run(&prog3).is_err());
    }
}