use crate::lang::*;

#[derive(Clone,Debug)]
pub struct SourceProgram {
    pub inputs: im::Vector<InputNode>,
    pub letBindings: im::Vector<LetNode>,
    pub expr: SourceExpr
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
pub struct InputNode(pub ArrayName, pub Shape);

impl Display for InputNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "input {}: {:?}", self.0, self.1)
    }
}

#[derive(Clone,Debug)]
pub struct LetNode(pub ArrayName, pub Box<SourceExpr>);

#[derive(Clone,Debug)]
pub enum SourceExpr {
    ForNode(IndexName, Extent, Box<SourceExpr>),
    ReduceNode(ExprOperator, Box<SourceExpr>),
    OpNode(ExprOperator, Box<SourceExpr>, Box<SourceExpr>),
    IndexingNode(ArrayName, im::Vector<IndexExpr>),
    LiteralNode(isize)
}

impl Display for SourceExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use SourceExpr::*;
        match self {
            ForNode(index, extent, body) => {
                write!(f, "for {} : {} in {}", index, extent, body)
            },

            ReduceNode(op, body) => {
                let reduce_op_str = 
                    match op {
                        ExprOperator::OpAdd => "sum",
                        ExprOperator::OpSub => "sum_sub",
                        ExprOperator::OpMul => "product"
                    };

                write!(f, "{}({})", reduce_op_str, body)
            },

            OpNode(op, expr1, expr2) => {
                write!(f, "({} {} {})", expr1, op, expr2)
            },

            IndexingNode(arr, index_list) => {
                write!(f, "{}{:?}", arr, index_list)
            },

            LiteralNode(val) => {
                write!(f, "{}", val)
            },
        }
    }
}

#[derive(Clone,Debug)]
pub enum IndexExpr {
    IndexVar(IndexName),
    IndexLiteral(isize),
    IndexOp(ExprOperator, Box<IndexExpr>, Box<IndexExpr>)
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
