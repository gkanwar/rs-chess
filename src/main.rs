use cairo;
use pixels::{Error, Pixels, SurfaceTexture};
use winit::dpi::{LogicalPosition};
use winit::event::MouseButton;
use winit::window::Window;
use std::f64::consts::PI;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Instant, Duration};
use winit::{
  dpi::LogicalSize,
  event::{ElementState, Event, WindowEvent},
  event_loop::{ControlFlow, EventLoop},
};
use rsvg::{Handle, HandleExt};
use rs_chess::game::*;


struct AppState {
  game_state: GameState,
  chess_set: ChessSet,
  selected_square: Option<Square>
}
impl AppState {
  fn handle_click(&mut self, square: Square) {
    let gs = &mut self.game_state;
    let content = gs.board.squares[square.0][square.1];
    match self.selected_square {
      None => {
        match content {
          SquareContent::Empty => {}
          SquareContent::Filled(piece) => {
            if piece.color == gs.active {
              self.selected_square = Some(square);
            }
          }
        }
      }
      Some(origin) => {
        // TODO: handle more complicated moves
        let proposed_move = Move::Normal {
          origin: origin, target: square
        };
        let bs2 = apply_move(&proposed_move, &gs.board);
        gs.board = bs2;
        self.selected_square = None;
      }
    }
  }
}

fn create_window(
  title: &str, w: u32, h: u32, scale: f64, event_loop: &EventLoop<()>,
) -> winit::window::Window {
  let size = LogicalSize::new((w as f64) * scale, (h as f64) * scale);
  let window = winit::window::WindowBuilder::new()
    .with_title(title)
    .with_inner_size(size)
    .with_resizable(false)
    .build(&event_loop)
    .unwrap();
  return window;
}

#[derive(Debug)]
struct ChessSet {
  white_king: Handle,
  black_king: Handle,
  white_queen: Handle,
  black_queen: Handle,
  white_rook: Handle,
  black_rook: Handle,
  white_bishop: Handle,
  black_bishop: Handle,
  white_knight: Handle,
  black_knight: Handle,
  white_pawn: Handle,
  black_pawn: Handle
}
impl ChessSet {
  pub fn cburnett() -> Self {
    ChessSet {
      white_king: Handle::new_from_data(include_bytes!("../media/cburnett/K.svg")).unwrap(),
      black_king: Handle::new_from_data(include_bytes!("../media/cburnett/k.svg")).unwrap(),
      white_queen: Handle::new_from_data(include_bytes!("../media/cburnett/Q.svg")).unwrap(),
      black_queen: Handle::new_from_data(include_bytes!("../media/cburnett/q.svg")).unwrap(),
      white_rook: Handle::new_from_data(include_bytes!("../media/cburnett/R.svg")).unwrap(),
      black_rook: Handle::new_from_data(include_bytes!("../media/cburnett/r.svg")).unwrap(),
      white_bishop: Handle::new_from_data(include_bytes!("../media/cburnett/B.svg")).unwrap(),
      black_bishop: Handle::new_from_data(include_bytes!("../media/cburnett/b.svg")).unwrap(),
      white_knight: Handle::new_from_data(include_bytes!("../media/cburnett/N.svg")).unwrap(),
      black_knight: Handle::new_from_data(include_bytes!("../media/cburnett/n.svg")).unwrap(),
      white_pawn: Handle::new_from_data(include_bytes!("../media/cburnett/P.svg")).unwrap(),
      black_pawn: Handle::new_from_data(include_bytes!("../media/cburnett/p.svg")).unwrap(),
    }
  }
}

fn render_drawing(ctx: &cairo::Context, handle: &Handle, x: f64, y: f64, dx: f64, dy: f64) {
  ctx.push_group();
  ctx.translate(x, y);
  let dims = handle.get_dimensions();
  ctx.scale(
      dx / dims.width as f64,
      dy / dims.height as f64
  );
  handle.render_cairo(&ctx);
  ctx.pop_group_to_source();
  ctx.paint();
}

fn render_piece(ctx: &cairo::Context, set: &ChessSet, piece: Piece, x: f64, y: f64, dx: f64, dy: f64) -> () {
  let handle = match (piece.kind, piece.color) {
    (PieceKind::King, Color::White) => &set.white_king,
    (PieceKind::King, Color::Black) => &set.black_king,
    (PieceKind::Queen, Color::White) => &set.white_queen,
    (PieceKind::Queen, Color::Black) => &set.black_queen,
    (PieceKind::Rook, Color::White) => &set.white_rook,
    (PieceKind::Rook, Color::Black) => &set.black_rook,
    (PieceKind::Bishop, Color::White) => &set.white_bishop,
    (PieceKind::Bishop, Color::Black) => &set.black_bishop,
    (PieceKind::Knight, Color::White) => &set.white_knight,
    (PieceKind::Knight, Color::Black) => &set.black_knight,
    (PieceKind::Pawn, Color::White) => &set.white_pawn,
    (PieceKind::Pawn, Color::Black) => &set.black_pawn,
  };
  render_drawing(ctx, handle, x, y, dx, dy);
}

fn render_full_board(ctx: &cairo::Context, state: &AppState) -> () {
  let (dx, dy) = (1.0 / (WIDTH as f64), 1.0 / (HEIGHT as f64));
  for i in 0..WIDTH {
    for j in 0..HEIGHT {
      let square: SquareContent = state.game_state.board.squares[i][j];
      if (i + j) % 2 == 0 {
        ctx.set_source_rgb(101.0/255.0, 138.0/255.0, 170.0/255.0);
      }
      else {
        ctx.set_source_rgb(183.0/255.0, 218.0/255.0, 234.0/255.0);
      }
      let (x, y) = ((i as f64) * dx, ((HEIGHT - j - 1) as f64) * dy);
      ctx.rectangle(x, y, dx, dy);
      ctx.fill();
      if let SquareContent::Filled(piece) = square {
        render_piece(ctx, &state.chess_set, piece, x, y, dx, dy);
      }
    }
  }
  if let Some((i, j)) = state.selected_square {
    let (x, y) = ((i as f64) * dx, ((HEIGHT - j - 1) as f64) * dy);
    ctx.set_line_width(0.1*dx);
    ctx.set_source_rgba(72.0/255.0, 77.0/255.0, 56.0/255.0, 0.5);
    ctx.arc(x+dx/2.0, y+dy/2.0, 0.45*dx, 0.0, 2.0*PI);
    ctx.stroke();
  }
}

enum EventMsg {
  Exit,
  Click {
    x: f64, y: f64
  }
}

fn game_thread(
  buffer_mutex: Arc<Mutex<Pixels>>, mut surface: cairo::ImageSurface,
  tx_render: mpsc::Sender<()>, rx_events: mpsc::Receiver<EventMsg>
) -> () {
  let gs: GameState =
    parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
  let mut state = AppState {
    game_state: gs,
    chess_set: ChessSet::cburnett(),
    selected_square: None
  };

  let render_to_buffer = |state: &AppState, surface: &mut cairo::ImageSurface| {
    let ctx: cairo::Context = cairo::Context::new(&surface);
    ctx.scale(surface.get_width() as f64, surface.get_height() as f64);
    render_full_board(&ctx, &state);
    drop(ctx);
    let mut buffer = buffer_mutex.lock().unwrap();
    buffer.frame_mut().copy_from_slice(&surface.get_data().unwrap());
  };
  render_to_buffer(&state, &mut surface);
  let _ = tx_render.send(());

  loop {
    let start = Instant::now();
    let end = start + Duration::from_millis(10);
    let recv_event = rx_events.try_recv();
    if let Ok(event) = recv_event {
      match event {
        EventMsg::Exit => {
          break;
        }
        EventMsg::Click { x, y } => {
          let (width, height) = (surface.get_width() as f64, surface.get_height() as f64);
          let sq_i = ((WIDTH as f64) * x / width) as usize;
          let sq_j = (HEIGHT-1) - ((HEIGHT as f64) * y / height) as usize;
          let square: Square = (sq_i, sq_j);
          state.handle_click(square);
          render_to_buffer(&state, &mut surface);
          let _ = tx_render.send(());
        }
      }
    }
    std::thread::sleep(end - start);
  }

  println!("Game thread end");
}

fn run_event_loop(
    event_loop: EventLoop<()>, buffer_mutex: Arc<Mutex<Pixels>>,
    rx_render: mpsc::Receiver<()>, tx_events: mpsc::Sender<EventMsg>, window: Window) -> ! {

  let scale = window.scale_factor();
  let mut mouse_pos: (f64, f64) = (-1.0, -1.0);
  const FPS: f64 = 60.0;
  const FRAME_US: u64 = (1000000.0 / FPS) as u64;

  event_loop.run(move |event, _, control_flow| {
    *control_flow = ControlFlow::Poll;
    control_flow.set_wait_until(Instant::now() + Duration::from_micros(FRAME_US));
    match event {
      Event::RedrawRequested(_) => {
        let buffer = buffer_mutex.lock().unwrap();
        let res = buffer.render();
        if res.map_err(|e| println!("err {}", e)).is_err() {
          *control_flow = ControlFlow::Exit;
        }
      }

      Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
        let _ = tx_events.send(EventMsg::Exit);
        *control_flow = ControlFlow::Exit;
      }

      Event::MainEventsCleared => {
        if let Ok(_) = rx_render.try_recv() {
          window.request_redraw();
        }
      }

      Event::WindowEvent {
        window_id: _,
        event: WindowEvent::CursorMoved {
          position, ..
        }
      } => {
        let LogicalPosition::<f64> {x, y} = position.to_logical(scale);
        mouse_pos = (x, y);
      }

      Event::WindowEvent {
        window_id: _,
        event: WindowEvent::MouseInput {
          state: ElementState::Pressed,
          button: MouseButton::Left, ..
        },
      } => {
        let _ = tx_events.send(EventMsg::Click {
          x: mouse_pos.0, y: mouse_pos.1
        });
      }
      _ => (),
    }
  });
}

fn main() -> Result<(), Error> {
  let event_loop = EventLoop::new();
  const WINDOW_WIDTH: u32 = 400;
  const WINDOW_HEIGHT: u32 = 400;
  let scale: f64 = 1.0;
  let window = create_window("gf-chess", WINDOW_WIDTH, WINDOW_HEIGHT, scale, &event_loop);
  let window_size = window.inner_size();

  let texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
  let buffer = Arc::new(Mutex::new(Pixels::new(WINDOW_WIDTH, WINDOW_HEIGHT, texture)?));

  let (tx_render, rx_render) = mpsc::channel::<()>();
  let (tx_events, rx_events) = mpsc::channel::<EventMsg>();

  let game_buffer = buffer.clone();
  thread::spawn(move || {
    let cairo_surface: cairo::ImageSurface =
      cairo::ImageSurface::create(
        cairo::Format::ARgb32, WINDOW_WIDTH as i32, WINDOW_HEIGHT as i32)
        .unwrap();
    game_thread(game_buffer, cairo_surface, tx_render, rx_events);
  });

  run_event_loop(event_loop, buffer, rx_render, tx_events, window);
}
