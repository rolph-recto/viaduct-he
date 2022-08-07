/// main.rs
/// Vectorizer for homomorphic encryption circuits

use clap::Parser;
use handlebars::{Handlebars, handlebars_helper};
use egg::{RecExpr, AstSize, CostFunction};
use crate::{lang::*, optimizer::optimize};

mod lang;
mod optimizer;

#[derive(Parser)]
#[clap(author, version, about = "optimizer for for vectorized homomorphic encryption circuits", long_about = None)]
struct Arguments {
    /// File to parse as input
    #[clap(value_parser)]
    file: String,

    #[clap(short = 'p', long = "template", value_parser, value_name = "template", default_value = "template.txt")]
    template: String,

    /// duration in seconds to run equality saturation until timeout
    #[clap(short = 'd', long = "duration", value_parser, value_name = "duration", default_value_t = 20)]
    duration: u64
}

handlebars_helper!(opfmt: |op: HEOperand| match op {
    HEOperand::NodeRef(r) => format!("i{}", r),
    HEOperand::ConstSym(sym) => format!("{}", sym),
    HEOperand::ConstNum(i) => format!("{}", i),
});

fn main() {
    let args = Arguments::parse();
    let input_str =
        std::fs::read_to_string(&args.file)
        .expect(&format!("Could not read file {}", &args.file));

    let template_str =
        std::fs::read_to_string(&args.template)
        .expect(&format!("Could not read file {}", &args.template));


    // parse the expression, the type annotation tells it which Language to use
    let init_expr: RecExpr<HE> = input_str.parse().unwrap();
    let init_prog = gen_program(&init_expr);
    let init_cost: usize = AstSize.cost_rec(&init_expr);

    println!("Running equality saturation for {} seconds...", args.duration);

    let (opt_cost, opt_expr ) = optimize(&init_expr, args.duration);
    let opt_prog: HEProgram = gen_program(&opt_expr);

    let mut handlebars = Handlebars::new();
    handlebars.register_helper("opfmt", Box::new(opfmt));
    handlebars.register_template_string("t", template_str)
        .expect("Could not register template");

    println!("Initial HE cost: {}", init_cost);
    println!("Initial HE expr:\n{}", init_expr.pretty(80));
    println!("Initial HE program:\n{}", handlebars.render("t", &init_prog).unwrap());
    
    println!("Optimized HE cost: {}", opt_cost);
    println!("Optimized HE expr:\n{}", opt_expr.pretty(80));
    println!("Optimized HE program:\n{}", handlebars.render("t", &opt_prog).unwrap());

    let vec_size = 16;
    let sym_store: HESymStore = init_prog.gen_sym_store(vec_size, -10..=10);
    let init_out = interp_program(&sym_store, &init_prog, vec_size);
    let opt_out = interp_program(&sym_store, &opt_prog, vec_size);

    // values of the last instructions should be equal
    println!("output for init prog: {}", init_out.unwrap());
    println!("output for opt prog: {}", opt_out.unwrap());
}
