#[cfg(test)]
mod tests {
    use he_vectorizer::lang::parser::ProgramParser;

    #[test]
    fn test_parse_positive() {
        let parser = ProgramParser::new();
        assert!(parser.parse("42").is_ok());
        assert!(parser.parse("(42)").is_ok());
        assert!(parser.parse("42 + 56").is_ok());
        assert!(parser.parse("42 * 56").is_ok());
        assert!(parser.parse("for x: 16 { 42 }").is_ok());
        assert!(parser
            .parse("for x: 16 { for y: 16 { img[x][y] + 2  }}")
            .is_ok());
        assert!(parser
            .parse(
                "
            let img2 = for x: 16 { for y: 16 { img[x][y] + 2  }} in
            img2 + img2
        "
            )
            .is_ok());
    }

    #[test]
    fn test_parse_negative() {
        let parser = ProgramParser::new();
        assert!(parser.parse("for x: 16 in { 42").is_err());
    }
}
