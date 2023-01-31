use crate::lang::*;

pub static OUTPUT_EXPR_NAME: &'static str = "$root";

#[derive(Clone,Debug)]
pub struct SourceProgram {
    input_map: HashMap<ArrayName, Shape>,
    expr_map: HashMap<ArrayName, SourceExpr>,
    pub inputs: im::Vector<Input>,
    pub let_bindings: im::Vector<LetBinding>,
    pub expr: SourceExpr
}

impl SourceProgram {
    pub fn new(inputs: im::Vector<Input>, let_bindings: im::Vector<LetBinding>, expr: SourceExpr) -> Self {
        // compute input map and expr binding map
        let mut input_map: HashMap<ArrayName, Shape> = HashMap::new();
        inputs.iter().for_each(|input| {
            if let Some(_) = input_map.insert(input.0.clone(), input.1.clone()) {
                panic!("duplicate bindings for {}", &input.0)
            }
        });

        let mut expr_binding_map: HashMap<ArrayName, SourceExpr> = HashMap::new();
        let_bindings.iter().for_each(|let_binding| {
            if let Some(_) = expr_binding_map.insert(let_binding.0.clone(), *let_binding.1.clone()) {
                panic!("duplicate bindings for {}", &let_binding.0)
            }
        });
        expr_binding_map.insert(String::from(OUTPUT_EXPR_NAME), expr.clone());

        SourceProgram {
            input_map, expr_map: expr_binding_map, inputs, let_bindings, expr
        }
    }

    pub fn get_input_shape(&self, array: &ArrayName) -> Option<&Shape> {
        self.input_map.get(array)
    }

    pub fn get_expr_binding(&self, array: &ArrayName) -> Option<&SourceExpr> {
        self.expr_map.get(array)
    }
}

impl Display for SourceProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inputs.iter().try_for_each(|input|
            write!(f, "{}", input)
        )?;
        write!(f, "{}", self.expr)
    }
}

#[derive(Clone,Debug)]
pub struct Input(pub ArrayName, pub Shape);

impl Display for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "input {}: {:?}", self.0, self.1)
    }
}

#[derive(Clone,Debug)]
pub struct LetBinding(pub ArrayName, pub Box<SourceExpr>);

#[derive(Clone,Debug)]
pub enum SourceExpr {
    For(IndexName, Extent, Box<SourceExpr>),
    Reduce(Operator, Box<SourceExpr>),
    ExprOp(Operator, Box<SourceExpr>, Box<SourceExpr>),
    Indexing(ArrayName, im::Vector<IndexExpr>),
    Literal(isize)
}

impl Display for SourceExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use SourceExpr::*;
        match self {
            For(index, extent, body) => {
                write!(f, "for {} : {} in {}", index, extent, body)
            },

            Reduce(op, body) => {
                let reduce_op_str = 
                    match op {
                        Operator::Add => "sum",
                        Operator::Sub => "sum_sub",
                        Operator::Mul => "product"
                    };

                write!(f, "{}({})", reduce_op_str, body)
            },

            ExprOp(op, expr1, expr2) => {
                write!(f, "({} {} {})", expr1, op, expr2)
            },

            Indexing(arr, index_list) => {
                write!(f, "{}{:?}", arr, index_list)
            },

            Literal(val) => {
                write!(f, "{}", val)
            },
        }
    }
}

#[derive(Clone,Debug)]
pub enum IndexExpr {
    IndexVar(IndexName),
    IndexLiteral(isize),
    IndexOp(Operator, Box<IndexExpr>, Box<IndexExpr>)
}

impl IndexExpr {
    pub fn get_single_var(&self) -> Option<IndexName> {
        let vars = self.get_vars();
        if vars.len() == 1 {
            vars.into_iter().last()

        } else {
            None
        }
    }

    pub fn get_vars(&self) -> im::HashSet<IndexName> {
        match self {
            IndexExpr::IndexVar(var) => im::HashSet::unit(var.clone()),

            IndexExpr::IndexLiteral(_) => im::HashSet::new(),

            IndexExpr::IndexOp(_, expr1, expr2) => {
                expr1.get_vars().union(expr2.get_vars())
            }
        }
    }
}

impl Display for IndexExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexExpr::IndexVar(var) => write!(f, "{}", var),

            IndexExpr::IndexLiteral(val) => write!(f, "{}", val),

            IndexExpr::IndexOp(op, expr1, expr2) => {
                write!(f, "({} {} {})", expr1, op, expr2)
            }
        }
    }
}
