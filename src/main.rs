extern crate lalrpop_util;

use std::{fs::File, io::Write, time::{Duration, Instant}};

/// main.rs
/// Vectorizer for homomorphic encryption circuits
use clap::Parser;
use log::*;

use he_vectorizer::{
    circ::{
        optimizer::{ExtractorType, Optimizer},
        materializer::{DefaultMaterializerFactory, MaterializerFactory},
        plaintext_hoisting::PlaintextHoisting, cost::CostFeatures,
    },
    lang::{
        index_elim::{IndexElimination, InlinedProgram},
        parser::ProgramParser,
        elaborated::Elaborator
    },
    scheduling::{
        scheduler::Scheduler,
        transformer::DefaultScheduleTransformerFactory,
    },
    program::{
        lowering::CircuitLowering,
        backend::{
            HEBackend,
            pyseal::SEALBackend, DummyBackend
        }
    },
};

#[derive(Parser)]
#[clap(author, version, about = "compiler for vectorized homomorphic encryption", long_about = None)]
struct HEArguments {
    /// file to parse as input
    #[clap(value_parser)]
    file: String,

    /// template file for output program
    #[clap(
        short = 't',
        long = "template",
        value_parser,
    )]
    template: Option<String>,

    /// file for output program
    #[clap(
        short = 'o',
        long = "outfile",
        value_parser,
    )]
    outfile: Option<String>,

    /// duration in seconds to run equality saturation (if 0, duration is unbounded)
    #[clap(short = 'd', long = "duration", value_parser, default_value_t = 20)]
    duration: usize,

    /// e-graph extractor to use
    #[clap(short = 'x', long = "extractor", value_enum, default_value_t = ExtractorType::LP)]
    extractor: ExtractorType,

    /// vector size
    #[clap(short = 's', long = "size", value_parser, default_value_t = 4096)]
    size: usize,

    /// don't inline instructions
    #[clap(short = 'n', long = "noinplace")]
    noinplace: bool,

    /// number of epochs to run scheduler
    #[clap(short = 'e', long = "epochs", value_parser, default_value_t = 1)]
    epochs: usize,

    /// number of epochs to run scheduler
    #[clap(short = 'b', long = "backend", value_parser, default_value = "pyseal")]
    backend: String,
}

fn get_backends(args: &HEArguments) -> Vec<Box<dyn HEBackend>> {
    vec![
        Box::new(DummyBackend),
        Box::new(
            SEALBackend::new(
                args.template.clone(), 
                !args.noinplace,
                args.size
            )
        ),
    ]
}

fn main() {
    let mut log_builder = env_logger::builder();
    log_builder.target(env_logger::Target::Stdout);
    log_builder.init();

    let args = HEArguments::parse();
    let input_str =
        std::fs::read_to_string(&args.file).expect(&format!("Could not read file {}", &args.file));

    info!("parsing...");
    let source = ProgramParser::new().parse(&input_str).unwrap();

    info!("elaboration...");
    let elaborated = Elaborator::new().run(source);

    info!("generating inline sets and array groups...");
    // let inline_sets = elaborated.simple_inline_sets();
    let inline_sets = vec![elaborated.no_inlined_set()];
    let inlined_programs: Vec<InlinedProgram> =
        inline_sets.into_iter().map(|inline_set| {
            let array_group = elaborated.array_group_from_inline_set(&inline_set);
            let inlined_program =
                IndexElimination::new()
                .run(&inline_set, &array_group, &elaborated)
                .unwrap();

            inlined_program
        }).collect();

    info!("scheduling...");
    let mut scheduler =
        Scheduler::new(
            inlined_programs,
            Box::new(DefaultScheduleTransformerFactory), 
            Box::new(DefaultMaterializerFactory), 
            args.size,
            args.epochs
        );

    scheduler.run(None);
    let best_opt = scheduler.get_best_schedule(CostFeatures::default_weights());

    if let None = best_opt {
        info!("No schedule found");
        return;
    }

    let (inlined, schedule, cost) = best_opt.unwrap();
    info!("found schedule:\n{}", schedule);
    info!("cost:\n{:?}", cost);
    info!("inlined program:\n{}", inlined);

    info!("circuit generation...");
    let materializer = DefaultMaterializerFactory.create();

    let res_materialize =
        materializer.run(&inlined, &schedule);

    let circuit = res_materialize.unwrap();
    info!("generated circuit:\n{}", circuit);

    let opt_circuit = 
        if args.duration > 0 {
            info!("circuit optimization...");
            let (opt_exprs, context) = circuit.to_opt_circuit();
            let (res_opt_exprs, opt_roots) =
                Optimizer::new(args.size)
                .optimize(opt_exprs, context, args.duration, args.extractor);

            let opt_circuit = circuit.from_opt_circuit(res_opt_exprs, opt_roots);
            opt_circuit

        } else {
            // run the optimizer even if there is no optimization
            // since the optimizer will perform value numbering
            let (opt_exprs, context) = circuit.to_opt_circuit();
            let (res_opt_exprs, opt_roots) =
                Optimizer::new(args.size)
                .optimize(opt_exprs, context, args.duration, args.extractor);

            circuit.from_opt_circuit(res_opt_exprs, opt_roots)
        };

    info!("plaintext hoisting...");
    let circuit_pe = PlaintextHoisting::new().run(opt_circuit);

    info!("circuit to lower:\n{}", circuit_pe);

    info!("circuit lowering...");
    let program = CircuitLowering::new().run(circuit_pe);
    info!("program:\n{}", program);

    let backends = get_backends(&args);

    let backend_opt =
        backends.into_iter()
        .find(|backend| backend.name() == args.backend);

    if let Some(mut backend) = backend_opt {
        info!("compiling with {} backend...", args.backend);

        let writer: Box<dyn std::io::Write> = 
            match &args.outfile {
                Some(outfile) => {
                    info!("wrote generated code to {}", outfile);
                    Box::new(File::create(outfile).unwrap())
                },

                None => {
                    info!("printed generated code to stdout");
                    Box::new(std::io::stdout())
                }
            };

        match backend.compile(program, writer) {
            Ok(_) => {},

            Err(err) => {
                println!("{}", err.to_string())
            }
        }

    } else {
        println!("no backend with name {} found", args.backend)
    }
}
