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

}

fn main() {
    let grid = Grid::new(5, 5, BoundaryMode::Wrap);
    println!("Grid initialized with {} cells!", grid.current.len());
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
