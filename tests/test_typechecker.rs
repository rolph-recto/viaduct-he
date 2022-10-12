use he_vectorizer::lang::{parser::ProgramParser, typechecker::TypeChecker};


#[test]
fn test_typechecker_positive() {
    let parser = ProgramParser::new();
    let typechecker = TypeChecker::new();

    let prog1 = parser.parse("42").unwrap();
    let prog2 = parser.parse("42 + 56").unwrap();
    let prog3 = parser.parse("
        input img: [(0,16),(0,16)]
        for x: (0,16) { img[x] }
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
        input img: [(0,16),(0,16)]
        for x: (0,16) {
            for y: (0, 16) {
                for z: (0, 16) { img[x][y][z] }
            }
        }
    ").unwrap();
    let prog3 = parser.parse("
        input img: [(0,16),(0,16)]
        let next = img + img
        for x: (0,16) {
            for y: (0, 16) {
                img[x][y] + next
            }
        }
    ").unwrap();

    assert!(typechecker.run(&prog1).is_err());
    assert!(typechecker.run(&prog2).is_err());
    assert!(typechecker.run(&prog3).is_err());
}