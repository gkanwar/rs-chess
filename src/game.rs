pub const WIDTH: usize = 8;
pub const HEIGHT: usize = 8;
pub const QROOK_FILE: usize = 0;
pub const KROOK_FILE: usize = WIDTH - 1;
pub const KING_FILE: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Color {
  Black,
  White,
}
pub fn opposite(color: Color) -> Color {
  match color {
    Color::Black => Color::White,
    Color::White => Color::Black,
  }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PieceKind {
  Pawn,
  Knight,
  Bishop,
  Rook,
  Queen,
  King,
}
#[derive(Clone, Copy, Debug)]
pub struct Piece {
  pub color: Color,
  pub kind: PieceKind,
}
#[derive(Clone, Copy, Debug)]
pub enum SquareContent {
  Empty,
  Filled(Piece),
}

#[derive(Clone, Debug)]
pub struct BoardState {
  pub squares: [[SquareContent; HEIGHT]; WIDTH],
}
impl BoardState {
  pub fn new() -> Self {
    BoardState {
      squares: [[SquareContent::Empty; HEIGHT]; WIDTH],
    }
  }
}

#[derive(Debug)]
pub struct CastlingState {
  wk: bool,
  wq: bool,
  bk: bool,
  bq: bool,
}
impl CastlingState {
  pub fn new() -> Self {
    CastlingState { wk: true, wq: true, bk: true, bq: true }
  }
}

#[derive(Debug)]
pub struct GameState {
  pub board: BoardState,
  pub castling: CastlingState,
  pub active: Color,
  pub en_passant: Option<Square>,
  // TODO: move counters
}
impl GameState {
  pub fn new() -> Self {
    Self {
      board: BoardState::new(),
      castling: CastlingState::new(),
      active: Color::White,
      en_passant: None
    }
  }
}

pub type Square = (usize, usize);

#[derive(Clone, Debug)]
pub enum Move {
  Normal {
    origin: Square,
    target: Square,
  },
  Promote {
    origin: Square,
    target: Square,
    kind: PieceKind,
  },
  CastleK(Color),
  CastleQ(Color),
}

fn in_bounds((x, y): (i32, i32)) -> bool {
  0 <= x && x < (WIDTH as i32) && 0 <= y && y < (HEIGHT as i32)
}

fn is_occupied((x, y): (i32, i32), board: &BoardState) -> bool {
  match board.squares[x as usize][y as usize] {
    SquareContent::Filled(_) => true, // TODO
    SquareContent::Empty => false,
  }
}

fn get_piece_pseudolegal_moves(
  (x, y): (usize, usize), color: Color, kind: PieceKind, board: &BoardState,
  en_passant: Option<Square>,
) -> Vec<Move> {
  let mut moves = Vec::<Move>::new();
  let (x, y) = (x as i32, y as i32);
  let en_passant = match en_passant {
    Some(square) => Some((square.0 as i32, square.1 as i32)),
    None => None,
  };

  let add_normal_move = |(xp, yp): (i32, i32), moves: &mut Vec<Move>| {
    moves.push(Move::Normal {
      origin: (x as usize, y as usize),
      target: (xp as usize, yp as usize),
    });
  };

  let check_square_piece = |(xp, yp): (i32, i32)| -> Option<Color> {
    if let SquareContent::Filled(piece) = board.squares[xp as usize][yp as usize] {
      return Some(piece.color);
    }
    return None;
  };

  // add square if valid, return whether space was free
  let maybe_add_square = |(xp, yp): (i32, i32), moves: &mut Vec<Move>| -> bool {
    match check_square_piece((xp, yp)) {
      Some(piece_color) => {
        if piece_color != color {
          add_normal_move((xp, yp), moves);
        }
        return false;
      }
      None => {
        add_normal_move((xp, yp), moves);
        return true;
      }
    }
  };

  let add_pawn_moves = |moves: &mut Vec<Move>| {
    let sign = match color {
      Color::White => 1,
      Color::Black => -1,
    };
    let queen_rank: i32 = match color {
      Color::White => (HEIGHT - 1) as i32,
      Color::Black => 0,
    };
    let start_rank: i32 = match color {
      Color::White => 1,
      Color::Black => (HEIGHT - 2) as i32,
    };

    let add_promo_moves = |origin: (i32, i32), target: (i32, i32), moves: &mut Vec<Move>| {
      for kind in &[PieceKind::Knight, PieceKind::Bishop, PieceKind::Rook, PieceKind::Queen] {
        moves.push(Move::Promote {
          origin: (origin.0 as usize, origin.1 as usize),
          target: (target.0 as usize, target.1 as usize),
          kind: *kind,
        });
      }
    };

    let push_one_sq = (x, y + sign);
    let push_two_sq = (x, y + 2 * sign);
    if in_bounds(push_one_sq) && !is_occupied(push_one_sq, board) {
      if y + sign != queen_rank {
        add_normal_move(push_one_sq, moves);
      }
      else {
        add_promo_moves((x, y), push_one_sq, moves);
      }
      if y == start_rank && !is_occupied(push_two_sq, board) {
        add_normal_move(push_two_sq, moves);
      }
    }
    // diagonal captures
    let diag_right_sq = (x + 1, y + sign);
    let diag_left_sq = (x - 1, y + sign);
    if in_bounds(diag_left_sq) {
      if Some(opposite(color)) == check_square_piece(diag_left_sq)
        || Some(diag_left_sq) == en_passant
      {
        if y + sign != queen_rank {
          add_normal_move(diag_left_sq, moves);
        }
        else {
          add_promo_moves((x, y), diag_left_sq, moves);
        }
      }
    }
    if in_bounds(diag_right_sq) {
      if Some(opposite(color)) == check_square_piece(diag_right_sq)
        || Some(diag_right_sq) == en_passant
      {
        if y + sign != queen_rank {
          add_normal_move(diag_right_sq, moves);
        }
        else {
          add_promo_moves((x, y), diag_right_sq, moves);
        }
      }
    }
  };

  let add_diagonal_moves = |moves: &mut Vec<Move>| {
    for xp in (x + 1)..(WIDTH as i32) {
      let yp = y - (x - xp);
      if !in_bounds((xp, yp)) {
        break;
      }
      if !maybe_add_square((xp, yp), moves) {
        break;
      }
    }
    for xp in (x + 1)..(WIDTH as i32) {
      let yp = y + (x - xp);
      if !in_bounds((xp, yp)) {
        break;
      }
      if !maybe_add_square((xp, yp), moves) {
        break;
      }
    }
    for xp in (0..x).rev() {
      let yp = y - (x - xp);
      if !in_bounds((xp, yp)) {
        break;
      }
      if !maybe_add_square((xp, yp), moves) {
        break;
      }
    }
    for xp in (0..x).rev() {
      let yp = y + (x - xp);
      if !in_bounds((xp, yp)) {
        break;
      }
      if !maybe_add_square((xp, yp), moves) {
        break;
      }
    }
  };

  let add_cardinal_moves = |moves: &mut Vec<Move>| {
    for xp in (x + 1)..(WIDTH as i32) {
      if !maybe_add_square((xp, y), moves) {
        break;
      }
    }
    for xp in (0..x).rev() {
      if !maybe_add_square((xp, y), moves) {
        break;
      }
    }
    for yp in (y + 1)..(WIDTH as i32) {
      if !maybe_add_square((x, yp), moves) {
        break;
      }
    }
    for yp in (0..y).rev() {
      if !maybe_add_square((x, yp), moves) {
        break;
      }
    }
  };

  let add_knight_moves = |moves: &mut Vec<Move>| {
    const KNIGHT_OFFSETS: &[(i32, i32)] =
      &[(-1, 2), (1, 2), (2, 1), (2, -1), (-1, -2), (1, -2), (-2, 1), (-2, -1)];
    for (dx, dy) in KNIGHT_OFFSETS.iter() {
      let square = (x + dx, y + dy);
      if in_bounds(square) {
        maybe_add_square(square, moves);
      }
    }
  };

  let add_king_moves = |moves: &mut Vec<Move>| {
    const KING_OFFSETS: &[(i32, i32)] =
      &[(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)];
    for (dx, dy) in KING_OFFSETS.iter() {
      let square = ((x as i32) + dx, (y as i32) + dy);
      if in_bounds(square) {
        maybe_add_square(square, moves);
      }
    }
  };

  match kind {
    PieceKind::Pawn => {
      add_pawn_moves(&mut moves);
    }
    PieceKind::Knight => {
      add_knight_moves(&mut moves);
    }
    PieceKind::Bishop => {
      add_diagonal_moves(&mut moves);
    }
    PieceKind::Rook => {
      add_cardinal_moves(&mut moves);
    }
    PieceKind::Queen => {
      add_diagonal_moves(&mut moves);
      add_cardinal_moves(&mut moves);
    }
    PieceKind::King => {
      add_king_moves(&mut moves);
    }
  }
  return moves;
}

fn apply_move(m: &Move, bs: &BoardState) -> BoardState {
  let mut bs2 = bs.clone();
  let squares = &mut bs2.squares;
  match m {
    Move::Normal { origin, target } => {
      squares[target.0][target.1] = squares[origin.0][origin.1];
      squares[origin.0][origin.1] = SquareContent::Empty;
    }
    Move::Promote { origin, target, kind } => {
      squares[target.0][target.1] = squares[origin.0][origin.1];
      if let SquareContent::Filled(piece) = squares[target.0][target.1] {
        squares[target.0][target.1] =
          SquareContent::Filled(Piece { color: piece.color, kind: *kind });
      }
      else {
        panic!("Promotion from an empty square");
      }
      squares[origin.0][origin.1] = SquareContent::Empty;
    }
    Move::CastleK(Color::White) => {
      assert_eq!(WIDTH, 8);
      let k = squares[KING_FILE][0];
      let r = squares[KROOK_FILE][0];
      squares[KROOK_FILE][0] = k;
      squares[KING_FILE][0] = r;
    }
    Move::CastleK(Color::Black) => {
      assert_eq!(WIDTH, 8);
      let k = squares[KING_FILE][HEIGHT - 1];
      let r = squares[KROOK_FILE][HEIGHT - 1];
      squares[KROOK_FILE][HEIGHT - 1] = k;
      squares[KING_FILE][HEIGHT - 1] = r;
    }
    Move::CastleQ(Color::White) => {
      assert_eq!(WIDTH, 8);
      let k = squares[KING_FILE][0];
      let r = squares[QROOK_FILE][0];
      squares[QROOK_FILE][0] = k;
      squares[KING_FILE][0] = r;
    }
    Move::CastleQ(Color::Black) => {
      assert_eq!(WIDTH, 8);
      let k = squares[KING_FILE][HEIGHT - 1];
      let r = squares[QROOK_FILE][HEIGHT - 1];
      squares[QROOK_FILE][HEIGHT - 1] = k;
      squares[KING_FILE][HEIGHT - 1] = r;
    }
  }
  return bs2;
}

fn is_capture(m: &Move, board: &BoardState) -> bool {
  match m {
    Move::Normal { origin: _, target } => {
      return match board.squares[target.0][target.1] {
        SquareContent::Filled(_) => true,
        SquareContent::Empty => false,
      };
    }
    _ => {
      return false;
    }
  }
}

fn get_pseudolegal_moves(
  board: &BoardState, en_passant: Option<Square>, active: Color,
) -> Vec<Move> {
  let mut moves = Vec::<Move>::new();
  for (i, row) in board.squares.iter().enumerate() {
    for (j, sq) in row.iter().enumerate() {
      if let SquareContent::Filled(Piece { color, kind }) = sq {
        if *color != active {
          continue;
        }
        moves.extend(
          get_piece_pseudolegal_moves((i, j), *color, *kind, &board, en_passant).iter().cloned(),
        );
      }
    }
  }
  return moves;
}

fn in_check(board: &BoardState, color: Color) -> bool {
  for m in get_pseudolegal_moves(board, None, opposite(color)) {
    match m {
      Move::Normal { origin, target } => {
        if let SquareContent::Filled(piece) = board.squares[target.0][target.1] {
          if piece.color == color && piece.kind == PieceKind::King {
            return true;
          }
        }
      }
      _ => {}
    }
  }
  return false;
}

fn get_legal_moves(gs: &GameState) -> Vec<Move> {
  let mut moves: Vec<Move> = get_pseudolegal_moves(&gs.board, gs.en_passant, gs.active)
    .iter()
    .filter(|m| {
      let bs2 = apply_move(m, &gs.board);
      return !in_check(&bs2, gs.active);
    })
    .cloned()
    .collect();
  let check_castle_ok = |rank: usize, rook_file: usize, king_file: usize| -> bool {
    let (king_target_file, rook_target_file, file1, file2) = if rook_file > king_file {
      (king_file + 2, king_file + 1, king_file, rook_file)
    }
    else {
      (king_file - 2, king_file - 1, rook_file, king_file)
    };
    // obstructed?
    for x in (file1 + 1)..file2 {
      if is_occupied((x as i32, rank as i32), &gs.board) {
        return false;
      }
    }
    // castling in/through check?
    let (file1, file2) = if king_target_file < king_file {
      (king_target_file, king_file)
    }
    else {
      (king_file, king_target_file)
    };
    for x in file1..(file2 + 1) {
      let mut bs2 = gs.board.clone();
      bs2.squares[x][rank] = bs2.squares[king_file][rank];
      bs2.squares[king_file][rank] = SquareContent::Empty;
      if in_check(&bs2, gs.active) {
        return false;
      }
    }
    return true;
  };
  // castling
  match gs.active {
    Color::White => {
      if gs.castling.wk && check_castle_ok(0, KROOK_FILE, KING_FILE) {
        moves.push(Move::CastleK(Color::White));
      }
      if gs.castling.wq && check_castle_ok(0, QROOK_FILE, KING_FILE) {
        moves.push(Move::CastleQ(Color::White));
      }
    }
    Color::Black => {
      if gs.castling.bk && check_castle_ok(HEIGHT - 1, KROOK_FILE, KING_FILE) {
        moves.push(Move::CastleK(Color::Black));
      }
      if gs.castling.bq && check_castle_ok(HEIGHT - 1, QROOK_FILE, KING_FILE) {
        moves.push(Move::CastleQ(Color::Black));
      }
    }
  }
  return moves;
}

fn get_fen_piece(piece_chr: &char) -> Piece {
  match piece_chr {
    'p' => Piece {
      color: Color::Black,
      kind: PieceKind::Pawn,
    },
    'r' => Piece {
      color: Color::Black,
      kind: PieceKind::Rook,
    },
    'n' => Piece {
      color: Color::Black,
      kind: PieceKind::Knight,
    },
    'b' => Piece {
      color: Color::Black,
      kind: PieceKind::Bishop,
    },
    'q' => Piece {
      color: Color::Black,
      kind: PieceKind::Queen,
    },
    'k' => Piece {
      color: Color::Black,
      kind: PieceKind::King,
    },
    'P' => Piece {
      color: Color::White,
      kind: PieceKind::Pawn,
    },
    'R' => Piece {
      color: Color::White,
      kind: PieceKind::Rook,
    },
    'N' => Piece {
      color: Color::White,
      kind: PieceKind::Knight,
    },
    'B' => Piece {
      color: Color::White,
      kind: PieceKind::Bishop,
    },
    'Q' => Piece {
      color: Color::White,
      kind: PieceKind::Queen,
    },
    'K' => Piece {
      color: Color::White,
      kind: PieceKind::King,
    },
    _ => {
      panic!("Unknown piece")
    }
  }
}

fn parse_fen_board(board_str: &str) -> Result<BoardState, &'static str> {
  let rows: Vec<_> = board_str.split("/").collect();
  if rows.len() != HEIGHT {
    return Err("FEN board state has wrong number of rows");
  }
  let mut board_state = BoardState::new();
  for (i, r) in rows.iter().enumerate() {
    let mut j = 0;
    for c in r.chars() {
      if "prnbqkPRNBQK".contains(c) {
        if j + 1 > WIDTH {
          return Err("Too many squares in FEN board row");
        }
        board_state.squares[j][HEIGHT - i - 1] = SquareContent::Filled(get_fen_piece(&c));
        j += 1;
      }
      else {
        let s: String = [c].iter().collect();
        match s.parse::<usize>() {
          Ok(num_spaces) => {
            if j + num_spaces > WIDTH {
              return Err("Too many squares in FEN board row");
            }
            for k in j..j + num_spaces {
              board_state.squares[k][HEIGHT - i - 1] = SquareContent::Empty;
            }
            j += num_spaces;
          }
          Err(_) => {
            return Err("Could not parse FEN board row");
          }
        }
      }
    }
  }
  return Ok(board_state);
}

fn parse_fen_castling(castle_str: &str) -> Result<CastlingState, &'static str> {
  let mut castling_state = CastlingState {
    wk: false,
    wq: false,
    bk: false,
    bq: false,
  };
  if castle_str == "-" {
    return Ok(castling_state);
  }
  for c in castle_str.chars() {
    match c {
      'K' => castling_state.wk = true,
      'Q' => castling_state.wq = true,
      'k' => castling_state.bk = true,
      'q' => castling_state.bq = true,
      _ => {
        return Err("Invalid FEN castling state");
      }
    }
  }
  return Ok(castling_state);
}

fn parse_square(sq_str: &str) -> Result<Square, &'static str> {
  if sq_str.len() < 2 {
    return Err("Invalid square string");
  }
  let col: char = sq_str.chars().next().unwrap().to_ascii_lowercase();
  if !col.is_ascii_alphabetic() {
    return Err("Invalid square column");
  }
  let mut buf = [0; 1];
  let col_i: usize = (col.encode_utf8(&mut buf).as_bytes()[0] - b'a').into();
  let row = &sq_str[1..];
  let row_i: usize;
  match row.parse::<usize>() {
    Ok(i) => {
      row_i = i - 1;
    }
    Err(_) => {
      return Err("Invalid square row");
    }
  }
  let square = (col_i, row_i);
  if !in_bounds((square.0 as i32, square.1 as i32)) {
    return Err("Square not in bounds");
  }
  return Ok(square);
}

fn parse_fen_en_passant(en_passant_str: &str) -> Result<Option<Square>, &'static str> {
  if en_passant_str == "-" {
    return Ok(None);
  }
  let sq = parse_square(en_passant_str)?;
  return Ok(Some(sq));
}

pub fn parse_fen(fen: &str) -> Result<GameState, &'static str> {
  let mut tok_iter = fen.split_whitespace();
  let mut game_state = GameState::new();

  let board_str = match tok_iter.next() {
    Some(s) => Ok(s),
    None => Err("FEN missing board string"),
  }?;
  game_state.board = parse_fen_board(board_str)?;

  let active = match tok_iter.next() {
    Some(s) => Ok(s),
    None => Err("FEN missing active string"),
  }?;
  match active {
    "w" => {
      game_state.active = Color::White;
    }
    "b" => {
      game_state.active = Color::Black;
    }
    _ => {
      return Err("FEN has invalid active color");
    }
  };

  let castle_str = match tok_iter.next() {
    Some(s) => Ok(s),
    None => Err("FEN missing castling string"),
  }?;
  game_state.castling = parse_fen_castling(castle_str)?;

  let en_passant_str = match tok_iter.next() {
    Some(s) => Ok(s),
    None => Err("FEN missing en passant string"),
  }?;
  game_state.en_passant = parse_fen_en_passant(en_passant_str)?;

  // TODO: halfmove clock?
  match tok_iter.next() {
    None => {
      return Ok(game_state);
    }
    _ => {}
  };
  match tok_iter.next() {
    None => {
      return Ok(game_state);
    }
    _ => {}
  };

  return Ok(game_state);
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_parse_fen() {
    parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
    parse_fen("rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2").unwrap();
  }

  #[test]
  fn test_legal_moves() {
    // position 1
    let gs: GameState =
      parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
    assert_eq!(get_legal_moves(&gs).len(), 20);

    // position 2
    let gs: GameState =
      parse_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -").unwrap();
    let moves = get_legal_moves(&gs);
    let captures = moves.iter().filter(|m| is_capture(m, &gs.board)).collect::<Vec<&Move>>();
    assert_eq!(moves.len(), 48);
    assert_eq!(captures.len(), 8);

    // position 3
    let gs: GameState = parse_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -").unwrap();
    let moves = get_legal_moves(&gs);
    let captures = moves.iter().filter(|m| is_capture(m, &gs.board)).collect::<Vec<&Move>>();
    assert_eq!(moves.len(), 14);
    assert_eq!(captures.len(), 1);

    // position 4
    let gs: GameState =
      parse_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1").unwrap();
    let moves = get_legal_moves(&gs);
    let captures = moves.iter().filter(|m| is_capture(m, &gs.board)).collect::<Vec<&Move>>();
    assert_eq!(moves.len(), 6);
    assert_eq!(captures.len(), 0);

    // position 5
    let gs: GameState =
      parse_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8").unwrap();
    let moves = get_legal_moves(&gs);
    println!("Moves: {:?}", moves);
    assert_eq!(moves.len(), 44);

    // position 6
    let gs: GameState =
      parse_fen("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10")
        .unwrap();
    let moves = get_legal_moves(&gs);
    println!("Moves: {:?}", moves);
    assert_eq!(moves.len(), 46);
  }
}