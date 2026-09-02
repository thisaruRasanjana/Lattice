use rand::RngExt;
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions};

#[derive(PartialEq)]
enum DrawModeState {
    Editing,
    Simulating,
}

#[allow(dead_code)]
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

fn get_index(width: usize, x: usize, y: usize) -> usize {
    y * width + x
}

fn live_neighbor_count(current: &[u8], width: usize, height: usize, boundary_mode: &BoundaryMode, x: usize, y: usize) -> u8 {
    let mut count = 0;
    let offsets: [(isize, isize); 8] = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1),
    ];

    for (dx, dy) in offsets {
        match boundary_mode {
            BoundaryMode::Wrap => {
                let nx = (x as isize + dx).rem_euclid(width as isize) as usize;
                let ny = (y as isize + dy).rem_euclid(height as isize) as usize;
                let index = get_index(width, nx, ny);
                count += current[index];
            }
            BoundaryMode::Fixed => {
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                if nx >= 0 && nx < width as isize && ny >= 0 && ny < height as isize {
                    let index = get_index(width, nx as usize, ny as usize);
                    count += current[index];
                }
            }
        }
    }
    count
}

fn update_cell(current: &[u8], width: usize, height: usize, boundary_mode: &BoundaryMode, x: usize, y: usize) -> u8 {
    let index = get_index(width, x, y);
    let alive = current[index] == 1;
    let neighbors = live_neighbor_count(current, width, height, boundary_mode, x, y);

    match (alive, neighbors) {
        (true, 2) | (true, 3) => 1,
        (false, 3) => 1,
        _ => 0,
    }
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
        get_index(self.width, x, y)
    }

    // 4. Update the grid to the next generation
    fn step(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width {
                let index = self.get_index(x, y);
                self.next[index] = update_cell(&self.current, self.width, self.height, &self.boundary_mode, x, y);
            }
        }

        // Swap the buffers in O(1) time! This prevents state contamination.
        std::mem::swap(&mut self.current, &mut self.next);
    }

    // 5. Update the grid in parallel using row-band partitioning
    fn step_parallel(&mut self, num_threads: usize) {
        let rows_per_chunk = (self.height + num_threads - 1) / num_threads;
        
        let width = self.width;
        let height = self.height;
        let boundary_mode = &self.boundary_mode;
        let current = &self.current;
        
        std::thread::scope(|s| {
            // chunks_mut automatically gives disjoint mutable slices of the backing array.
            for (chunk_idx, next_chunk) in self.next.chunks_mut(rows_per_chunk * width).enumerate() {
                s.spawn(move || {
                    let start_y = chunk_idx * rows_per_chunk;
                    let chunk_height = next_chunk.len() / width;
                    
                    for dy in 0..chunk_height {
                        let y = start_y + dy;
                        for x in 0..width {
                            let next_state = update_cell(current, width, height, boundary_mode, x, y);
                            next_chunk[dy * width + x] = next_state;
                        }
                    }
                });
            }
        });

        std::mem::swap(&mut self.current, &mut self.next);
    }

    // Helper to visualize the grid in the terminal
    #[allow(dead_code)]
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
    } else if mode == "par" {
        let arg2 = args.get(2).map(|s| s.as_str()).unwrap_or("sweep");
        let width = args.get(3).and_then(|v| v.parse::<usize>().ok()).unwrap_or(512);
        let generations = args.get(4).and_then(|v| v.parse::<usize>().ok()).unwrap_or(100);

        println!("Lattice — Phase 2: Multi-threaded CPU Benchmark");
        println!("──────────────────────────────────────────");
        println!("Grid size:       {} × {} ({} cells)", width, width, width * width);
        println!("Generations:     {}", generations);
        println!();

        if arg2 == "sweep" {
            println!("{:<8} | {:<12} | {:<16} | {:<9} | {:<10}", "Threads", "Total Time", "Throughput", "Speedup", "Est. BW");
            println!("─────────|──────────────|──────────────────|───────────|──────────");
            
            let thread_counts = vec![1, 2, 4, 8, 10, 16];
            let mut baseline_time = 0.0;
            
            for t in thread_counts {
                let mut grid = Grid::new(width, width, BoundaryMode::Wrap);
                grid.randomize();
                
                let start = std::time::Instant::now();
                for _ in 0..generations {
                    grid.step_parallel(t);
                }
                let elapsed = start.elapsed();
                let elapsed_secs = elapsed.as_secs_f64();
                
                if t == 1 {
                    baseline_time = elapsed_secs;
                }
                
                let total_cells = (width * width * generations) as f64;
                let cells_per_sec = total_cells / elapsed_secs;
                let speedup = baseline_time / elapsed_secs;
                let est_bandwidth = (total_cells * 10.0 / elapsed_secs) / 1_000_000_000.0;
                
                println!("{:<8} | {:<10.2?} | {:>6.2} M c/s    | {:>5.2}x   | {:>5.2} GB/s", 
                    t, elapsed, cells_per_sec / 1_000_000.0, speedup, est_bandwidth);
            }
            println!("──────────────────────────────────────────");
        } else {
            let num_threads = arg2.parse::<usize>().unwrap_or(4);
            let mut grid = Grid::new(width, width, BoundaryMode::Wrap);
            grid.randomize();
            
            let start = std::time::Instant::now();
            for _ in 0..generations {
                grid.step_parallel(num_threads);
            }
            let elapsed = start.elapsed();
            
            let total_cells = (width * width * generations) as f64;
            let elapsed_secs = elapsed.as_secs_f64();
            let cells_per_sec = total_cells / elapsed_secs;
            let est_bandwidth_gb_s = (total_cells * 10.0 / elapsed_secs) / 1_000_000_000.0;

            println!("Threads:         {}", num_threads);
            println!("Total time:      {:.2?}", elapsed);
            println!("Throughput:      {:.2} M cells/sec", cells_per_sec / 1_000_000.0);
            println!("Est. bandwidth:  ~{:.2} GB/s", est_bandwidth_gb_s);
            println!("──────────────────────────────────────────");
        }
    } else {
        // Sequential CPU Benchmark Mode
        let (size, generations) = if mode == "seq" {
            let s = args.get(2).and_then(|v| v.parse::<usize>().ok()).unwrap_or(512);
            let g = args.get(3).and_then(|v| v.parse::<usize>().ok()).unwrap_or(100);
            (s, g)
        } else if let Ok(s) = mode.parse::<usize>() {
            let g = args.get(2).and_then(|v| v.parse::<usize>().ok()).unwrap_or(100);
            (s, g)
        } else {
            (512, 100)
        };

        let width = size;
        let height = size;
        let mut grid = Grid::new(width, height, BoundaryMode::Wrap);
        grid.randomize();

        println!("Lattice — Phase 1: Sequential CPU Benchmark");
        println!("──────────────────────────────────────────");
        println!("Grid size:       {} × {} ({} cells)", width, height, width * height);
        println!("Generations:     {}", generations);
        println!("Seeding grid...  ready.");
        println!("Running benchmark...");

        let start = std::time::Instant::now();
        for _ in 0..generations {
            grid.step();
        }
        let elapsed = start.elapsed();

        let total_cells_processed = (width * height * generations) as f64;
        let elapsed_secs = elapsed.as_secs_f64();
        let cells_per_sec = total_cells_processed / elapsed_secs;
        let time_per_gen_ms = (elapsed.as_secs_f64() * 1000.0) / (generations as f64);

        // Memory bandwidth estimation:
        let bytes_per_cell = 10.0;
        let total_bytes_accessed = total_cells_processed * bytes_per_cell;
        let est_bandwidth_gb_s = (total_bytes_accessed / elapsed_secs) / 1_000_000_000.0;

        println!("──────────────────────────────────────────");
        println!("Total time:      {:.2?}", elapsed);
        println!("Time / gen:      {:.3} ms", time_per_gen_ms);
        println!("Throughput:      {:.2} M cells/sec", cells_per_sec / 1_000_000.0);
        println!("Est. memory:     ~{:.2} GB accessed", total_bytes_accessed / 1_000_000_000.0);
        println!("Est. bandwidth:  ~{:.2} GB/s", est_bandwidth_gb_s);
        println!("──────────────────────────────────────────");
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

    #[test]
    fn test_parallel_correctness() {
        // Create two identical randomized grids
        let mut grid_seq = Grid::new(100, 100, BoundaryMode::Wrap);
        grid_seq.randomize();
        
        let mut grid_par = Grid::new(100, 100, BoundaryMode::Wrap);
        grid_par.current.copy_from_slice(&grid_seq.current);

        // Run both for 10 generations
        for _ in 0..10 {
            grid_seq.step();
            grid_par.step_parallel(4);
        }

        // Assert they are byte-for-byte identical
        assert_eq!(grid_seq.current, grid_par.current);
    }
}
