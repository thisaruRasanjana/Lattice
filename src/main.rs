use rand::RngExt;
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions};

#[derive(PartialEq)]
enum DrawModeState {
    Editing,
    Simulating,
}

enum BoundaryMode{
    Wrap,
    Fixed,
}

struct Grid {
    width: usize,
    height: usize,
    boundary_mode: BoundaryMode,
    current: Vec<u8>,
    next: Vec<u8>,
}

impl Grid {
    // 1. Initialize the grid
    fn new(width: usize, height: usize, boundary_mode: BoundaryMode) -> Self {
        let total_size = width * height;
        
        Grid {
            width,
            height,
            boundary_mode,
            // vec![0; size] creates an array of zeros of that exact size
            current: vec![0; total_size],
            next: vec![0; total_size],
        }
    }

    // 2. The 2D to 1D translation formula
    fn get_index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

        fn live_neighbor_count(&self, x: usize, y: usize) -> u8 {
        let mut count = 0;

        // The 8 directions around a cell (dx, dy)
        let offsets: [(isize, isize); 8] = [
            (-1, -1), (0, -1), (1, -1), // Top row
            (-1,  0),          (1,  0), // Middle row
            (-1,  1), (0,  1), (1,  1), // Bottom row
        ];

        for (dx, dy) in offsets {
            match self.boundary_mode {
                BoundaryMode::Wrap => {
                    // 1. Calculate the raw neighbor coordinates
                    let nx = (x as isize + dx).rem_euclid(self.width as isize) as usize;
                    let ny = (y as isize + dy).rem_euclid(self.height as isize) as usize;

                    // 2. Convert to index and add to `count`
                    let index = self.get_index(nx, ny);
                    count += self.current[index];

                }
                BoundaryMode::Fixed => {
                    // 1. Calculate the raw neighbor coordinates
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;

                    // 2. Check if the neighbor is within bounds
                    if nx >= 0 && nx < self.width as isize && ny >= 0 && ny < self.height as isize {
                        let index = self.get_index(nx as usize, ny as usize);
                        count += self.current[index];
                    }
                }
            }
        }

        count
    }

    // 4. Update the grid to the next generation
    fn step(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width {
                let index = self.get_index(x, y);
                let alive = self.current[index] == 1;
                let neighbors = self.live_neighbor_count(x, y);

                let next_state = match (alive, neighbors) {
                    // Rule 1: Any live cell with 2 or 3 live neighbours survives
                    (true, 2) | (true, 3) => 1,
                    // Rule 2: Any dead cell with exactly 3 live neighbours becomes live
                    (false, 3) => 1,
                    // Rule 3: All other cells die or stay dead
                    _ => 0,
                };

                self.next[index] = next_state;
            }
        }

        // Swap the buffers in O(1) time! This prevents state contamination.
        std::mem::swap(&mut self.current, &mut self.next);
    }

    // Helper to visualize the grid in the terminal
    fn print(&self) {
        for y in 0..self.height {
            for x in 0..self.width {
                let index = self.get_index(x, y);
                let symbol = if self.current[index] == 1 { "█ " } else { ". " };
                print!("{}", symbol);
            }
            println!();
        }
    }

    // Randomize the grid with a roughly 20% fill rate
    fn randomize(&mut self) {
        let mut rng = rand::rng(); // rand v0.9+ syntax
        for i in 0..self.current.len() {
            if rng.random_ratio(1, 5) {
                self.current[i] = 1;
            } else {
                self.current[i] = 0;
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = if args.len() > 1 { args[1].as_str() } else { "seq" };

    if mode == "visual" || mode == "draw" {
        let is_draw = mode == "draw";
        let width = if is_draw { 50 } else { 100 };
        let height = if is_draw { 50 } else { 100 };
        let scale = if is_draw { minifb::Scale::X16 } else { minifb::Scale::X8 };

        let mut grid = Grid::new(width, height, BoundaryMode::Wrap);
        if !is_draw {
            grid.randomize();
        }

        let title = if is_draw {
            "Lattice - Draw Mode | Space to Start | R to Reset"
        } else {
            "Lattice - Game of Life"
        };

        let mut window = Window::new(
            title,
            width,
            height,
            WindowOptions {
                scale,
                ..WindowOptions::default()
            },
        ).unwrap_or_else(|e| panic!("{}", e));

        // Use 60 FPS for responsive drawing, 8 FPS for simulation
        let mut current_fps = if is_draw { 60 } else { 8 };
        window.set_target_fps(current_fps);

        let mut buffer: Vec<u32> = vec![0; width * height];
        let mut state = if is_draw { DrawModeState::Editing } else { DrawModeState::Simulating };
        let mut last_click_pos: Option<(usize, usize)> = None;
        
        while window.is_open() && !window.is_key_down(Key::Escape) {
            // Handle Reset
            if window.is_key_down(Key::R) {
                grid.current.fill(0);
                if is_draw {
                    state = DrawModeState::Editing;
                    window.set_title("Lattice - Draw Mode | Space to Start | R to Reset");
                    current_fps = 60;
                    window.set_target_fps(current_fps);
                }
            }

            match state {
                DrawModeState::Editing => {
                    if window.is_key_pressed(Key::Space, KeyRepeat::No) {
                        state = DrawModeState::Simulating;
                        window.set_title("Lattice - Simulating | R to Reset");
                        current_fps = 8;
                        window.set_target_fps(current_fps);
                    }

                    if let Some((px, py)) = window.get_mouse_pos(MouseMode::Discard) {
                        let tx = px as usize;
                        let ty = py as usize;
                        if tx < width && ty < height {
                            let left_down = window.get_mouse_down(MouseButton::Left);
                            let right_down = window.get_mouse_down(MouseButton::Right);

                            if left_down || right_down {
                                let current_pos = (tx, ty);
                                if last_click_pos != Some(current_pos) {
                                    let index = grid.get_index(tx, ty);
                                    if left_down {
                                        grid.current[index] = 1;
                                    } else {
                                        grid.current[index] = 0;
                                    }
                                    last_click_pos = Some(current_pos);
                                }
                            } else {
                                last_click_pos = None;
                            }
                        }
                    }
                }
                DrawModeState::Simulating => {
                    grid.step();
                }
            }

            for (i, cell) in grid.current.iter().enumerate() {
                buffer[i] = if *cell == 1 { 0x00_00_FF_88 } else { 0x00_11_11_11 };
            }

            window.update_with_buffer(&buffer, width, height).unwrap();
        }
    } else {
        // Fallback to the headless glider test
        let mut grid = Grid::new(10, 10, BoundaryMode::Wrap);
        
        let glider_coords = [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)];
        for (x, y) in glider_coords {
            let index = grid.get_index(x, y);
            grid.current[index] = 1;
        }

        println!("Generation 0:");
        grid.print();

        for i in 1..=5 {
            grid.step();
            println!("\nGeneration {}:", i);
            grid.print();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_index() {
        let grid = Grid::new(5, 5, BoundaryMode::Wrap);
        
        // Top-left corner
        assert_eq!(grid.get_index(0, 0), 0);
        
        // Start of the second row (skipped one full row of 5)
        assert_eq!(grid.get_index(0, 1), 5);
        
        // Middle of the grid (Row 2, Column 3) -> (2 * 5) + 3 = 13
        assert_eq!(grid.get_index(3, 2), 13);
        
        // Bottom-right corner (Row 4, Column 4) -> (4 * 5) + 4 = 24
        assert_eq!(grid.get_index(4, 4), 24);
    }
}
