use std::collections::HashSet;

use indexmap::IndexMap;

use crate::lang::*;

#[derive(Clone, Debug)]
pub struct SourceProgram {
    // use IndexMap instead of HashMap to preserve the program order
    pub input_map: IndexMap<ArrayName, (Shape, ArrayType)>,
    pub expr_map: IndexMap<ArrayName, SourceExpr>,
}

impl SourceProgram {
    pub fn new(
        inputs: im::Vector<Input>,
        let_bindings: im::Vector<LetBinding>,
        output_expr: SourceExpr,
    ) -> Self {
        // compute input map and expr binding map
        let mut input_map: IndexMap<ArrayName, (Shape, ArrayType)> = IndexMap::new();
        inputs.iter().for_each(|input| {
            if let Some(_) = input_map.insert(input.0.clone(), (input.1.clone(), input.2)) {
                panic!("duplicate bindings for {}", &input.0)
            }
        });

        let mut expr_map: IndexMap<ArrayName, SourceExpr> = IndexMap::new();
        let_bindings.iter().for_each(|let_binding| {
            if let Some(_) = expr_map.insert(let_binding.0.clone(), *let_binding.1.clone()) {
                panic!("duplicate bindings for {}", &let_binding.0)
            }
        });
        expr_map.insert(String::from(OUTPUT_EXPR_NAME), output_expr.clone());

        SourceProgram {
            input_map,
            expr_map,
        }
    }

    pub fn is_input(&self, array: &String) -> bool {
        self.input_map.contains_key(array)
    }

    pub fn is_expr(&self, array: &String) -> bool {
        self.expr_map.contains_key(array)
    }

    pub fn get_output_expr(&self) -> &SourceExpr {
        self.expr_map.get(OUTPUT_EXPR_NAME).unwrap()
    }

    pub fn get_input_shape(&self, array: &ArrayName) -> Option<&Shape> {
        self.input_map.get(array).map(|(shape, _)| shape)
    }

    pub fn get_expr(&self, array: &ArrayName) -> Option<&SourceExpr> {
        self.expr_map.get(array)
    }
}

impl Display for SourceProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.input_map
            .iter()
            .try_for_each(|(array, shape)| write!(f, "{}{:?}", array, shape))?;
        self.expr_map
            .iter()
            .try_for_each(|(array, expr)| write!(f, "let {} = {}", array, expr))?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Input(pub ArrayName, pub Shape, pub ArrayType);

impl Display for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "input {}: {:?}", self.0, self.1)
    }
}

#[derive(Clone, Debug)]
pub struct LetBinding(pub ArrayName, pub Box<SourceExpr>);

#[derive(Clone, Debug)]
pub enum SourceExpr {
    For(IndexVar, Extent, Box<SourceExpr>),
    Reduce(Operator, Box<SourceExpr>),
    ExprOp(Operator, Box<SourceExpr>, Box<SourceExpr>),
    Indexing(ArrayName, im::Vector<IndexExpr>),
    Literal(isize),
}

impl SourceExpr {
    pub fn get_indexed_arrays(&self) -> HashSet<String> {
        match self {
            SourceExpr::For(_, _, body) => body.get_indexed_arrays(),

            SourceExpr::Reduce(_, body) => body.get_indexed_arrays(),

            SourceExpr::ExprOp(_, expr1, expr2) => {
                let mut arrays1 = expr1.get_indexed_arrays();
                let mut arrays2 = expr2.get_indexed_arrays();
                arrays1.extend(arrays2);
                arrays1
            }

            SourceExpr::Indexing(array, _) => HashSet::from([array.clone()]),

            SourceExpr::Literal(_) => HashSet::new(),
        }
    }
}

impl Display for SourceExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use SourceExpr::*;
        match self {
            For(index, extent, body) => {
                write!(f, "for {} : {} in {}", index, extent, body)
            }

            Reduce(op, body) => {
                let reduce_op_str = match op {
                    Operator::Add => "sum",
                    Operator::Sub => "sum_sub",
                    Operator::Mul => "product",
                };

                write!(f, "{}({})", reduce_op_str, body)
            }

            ExprOp(op, expr1, expr2) => {
                write!(f, "({} {} {})", expr1, op, expr2)
            }

            Indexing(arr, index_list) => {
                let mut index_str = String::new();
                for index in index_list.iter() {
                    index_str.push_str(&format!("[{}]", index))
                }
                write!(f, "{}{}", arr, index_str)
            }

            Literal(val) => {
                write!(f, "{}", val)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum IndexExpr {
    Var(IndexVar),
    Literal(isize),
    Op(Operator, Box<IndexExpr>, Box<IndexExpr>),
}

impl IndexExpr {
    pub fn get_single_var(&self) -> Option<IndexVar> {
        let vars = self.get_vars();
        if vars.len() == 1 {
            vars.into_iter().last()
        } else {
            None
        }
    }

    pub fn get_vars(&self) -> im::HashSet<IndexVar> {
        match self {
            IndexExpr::Var(var) => im::HashSet::unit(var.clone()),

            IndexExpr::Literal(_) => im::HashSet::new(),

            IndexExpr::Op(_, expr1, expr2) => expr1.get_vars().union(expr2.get_vars()),
        }
    }
}

impl Display for IndexExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexExpr::Var(var) => write!(f, "{}", var),

            IndexExpr::Literal(val) => write!(f, "{}", val),

            IndexExpr::Op(op, expr1, expr2) => {
                write!(f, "({} {} {})", expr1, op, expr2)
            }
        }
    }
}
