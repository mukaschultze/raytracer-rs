// https://www.gabrielgambetta.com/computer-graphics-from-scratch/basic-ray-tracing.html

extern crate png;

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const BACKGROUND_COLOR: (u8, u8, u8, u8) = (0, 0, 0, 0);

struct Scene {
    camera_pos: (f64, f64, f64),
    camera_d: f64,
    viewport_size: (f64, f64),
    spheres: Vec<Sphere>,
    lights: Vec<Light>,
}

struct Sphere {
    center: (f64, f64, f64),
    radius: f64,
    color: (u8, u8, u8),
}

struct Texture {
    width: usize,
    height: usize,
    data: Vec<u8>,
}

enum Light {
    Ambient {
        intensity: f64,
    },
    Point {
        intensity: f64,
        position: (f64, f64, f64),
    },
    Directional {
        intensity: f64,
        direction: (f64, f64, f64),
    },
}

impl Texture {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0u8; width * height * 4],
        }
    }

    pub fn screenshot(&self, path: &str) {
        println!("Writing file {}", path);
        let file = File::create(Path::new(path)).unwrap();
        let ref mut w = BufWriter::new(file);
        let mut encoder = png::Encoder::new(w, self.width as u32, self.height as u32);
        encoder.set_color(png::ColorType::RGBA);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::Default);
        encoder.set_filter(png::FilterType::NoFilter);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&self.data[..]).unwrap();
    }

    // pub fn put_pixel_rgb(&mut self, (x, y): (usize, usize), (r, g, b): (u8, u8, u8)) {
    //     self.put_pixel_rgba((x, y), (r, g, b, 255))
    // }

    pub fn put_pixel_rgba(&mut self, (x, y): (usize, usize), (r, g, b, a): (u8, u8, u8, u8)) {
        let idx = (y * self.width + x) * 4;
        self.data[idx + 0] = r;
        self.data[idx + 1] = g;
        self.data[idx + 2] = b;
        self.data[idx + 3] = a;
    }
}

pub fn main() {
    let mut texture = Texture::new(512, 512);

    let scene = Scene {
        camera_pos: (0.0, 0.0, 0.0), // O
        camera_d: 1.0,               // d, Distance from camera to viewport
        viewport_size: (1.0, 1.0),   // vw, vh
        spheres: vec![
            Sphere {
                center: (0.0, -1.0, 3.0),
                radius: 1.0,
                color: (255, 0, 0), // Red
            },
            Sphere {
                center: (2.0, 0.0, 4.0),
                radius: 1.0,
                color: (0, 0, 255), // Blue
            },
            Sphere {
                center: (-2.0, 0.0, 4.0),
                radius: 1.0,
                color: (0, 255, 0), // Green
            },
            Sphere {
                color: (255, 255, 0), // Yellow
                center: (0.0, -5001.0, 0.0),
                radius: 5000.0,
            },
        ],
        lights: vec![
            Light::Ambient { intensity: 0.2 },
            Light::Point {
                intensity: 0.6,
                position: (2.0, 1.0, 0.0),
            },
            Light::Directional {
                intensity: 0.2,
                direction: (1.0, 4.0, 4.0),
            },
        ],
    };

    for cy in -(texture.height as i32 / 2)..(texture.height as i32 / 2) {
        for cx in -(texture.width as i32 / 2)..(texture.width as i32 / 2) {
            let d = canvas_to_viewport(
                (cx as f64, cy as f64),
                (texture.width as f64, texture.height as f64),
                scene.viewport_size,
                scene.camera_d,
            );
            let rgba = trace_ray(&scene, d, 1.0, std::f64::INFINITY);

            texture.put_pixel_rgba(
                (
                    (cx + (texture.width as i32 / 2)) as usize,
                    texture.height - 1 - (cy + (texture.height as i32 / 2)) as usize,
                ),
                rgba,
            );
        }
    }

    texture.screenshot("output.png");
}

fn between<T>(value: T, min: T, max: T) -> bool
where
    T: std::cmp::PartialOrd,
{
    value >= min && value <= max
}

fn trace_ray(scene: &Scene, d: (f64, f64, f64), t_min: f64, t_max: f64) -> (u8, u8, u8, u8) {
    let o = scene.camera_pos;
    let mut closest_t = std::f64::INFINITY;
    let mut closest_sphere = None;

    for sphere in &scene.spheres {
        let (t1, t2) = intersect_ray_sphere(o, d, sphere);

        if between(t1, t_min, t_max) && t1 < closest_t {
            closest_t = t1;
            closest_sphere = Some(sphere);
        }

        if between(t2, t_min, t_max) && t2 < closest_t {
            closest_t = t2;
            closest_sphere = Some(sphere);
        }
    }

    match closest_sphere {
        None => BACKGROUND_COLOR,
        Some(sphere) => {
            let p = add(o, mul_tuple(d, closest_t)); // Compute intersection
            let n = sub(p, sphere.center); // Compute sphere normal at intersection
            let n = div_tuple(n, length(n)); // Normalize

            let lighting = compute_lighting(scene, p, n);
            let (r, g, b) = sphere.color;

            (
                (r as f64 * lighting) as u8,
                (g as f64 * lighting) as u8,
                (b as f64 * lighting) as u8,
                255,
            )
        }
    }
}

fn dot<T>((a1, a2, a3): (T, T, T), (b1, b2, b3): (T, T, T)) -> T
where
    T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    a1 * b1 + a2 * b2 + a3 * b3
}

fn sub<T>((a1, a2, a3): (T, T, T), (b1, b2, b3): (T, T, T)) -> (T, T, T)
where
    T: std::ops::Sub<Output = T>,
{
    (a1 - b1, a2 - b2, a3 - b3)
}

fn add<T>((a1, a2, a3): (T, T, T), (b1, b2, b3): (T, T, T)) -> (T, T, T)
where
    T: std::ops::Add<Output = T>,
{
    (a1 + b1, a2 + b2, a3 + b3)
}

fn mul<T>((a1, a2, a3): (T, T, T), (b1, b2, b3): (T, T, T)) -> (T, T, T)
where
    T: std::ops::Mul<Output = T>,
{
    (a1 * b1, a2 * b2, a3 * b3)
}

fn mul_tuple<T>((a1, a2, a3): (T, T, T), b: T) -> (T, T, T)
where
    T: std::ops::Mul<Output = T> + Copy,
{
    (a1 * b, a2 * b, a3 * b)
}

fn div_tuple<T>((a1, a2, a3): (T, T, T), b: T) -> (T, T, T)
where
    T: std::ops::Div<Output = T> + Copy,
{
    (a1 / b, a2 / b, a3 / b)
}

fn length((x, y, z): (f64, f64, f64)) -> f64 {
    (x.powi(2) + y.powi(2) + z.powi(2)).sqrt()
}

fn intersect_ray_sphere(o: (f64, f64, f64), d: (f64, f64, f64), sphere: &Sphere) -> (f64, f64) {
    let c = sphere.center;
    let r = sphere.radius;
    let oc = sub(o, c);

    let k1 = dot(d, d);
    let k2 = 2.0 * dot(oc, d);
    let k3 = dot(oc, oc) - r * r;

    let discriminant = k2 * k2 - 4.0 * k1 * k3;

    if discriminant < 0.0 {
        (std::f64::INFINITY, std::f64::INFINITY)
    } else {
        (
            (-k2 + discriminant.sqrt()) / (2.0 * k1),
            (-k2 - discriminant.sqrt()) / (2.0 * k1),
        )
    }
}

fn compute_lighting(scene: &Scene, p: (f64, f64, f64), n: (f64, f64, f64)) -> f64 {
    let mut i = 0.0;

    for light in &scene.lights {
        match light {
            Light::Ambient { intensity } => {
                i += intensity;
            }
            Light::Point {
                intensity,
                position,
            } => {
                let l = sub(*position, p);
                let n_dot_l = dot(n, l);

                if n_dot_l > 0.0 {
                    i += intensity * n_dot_l / (length(n) * length(l))
                }
            }
            Light::Directional {
                intensity,
                direction,
            } => {
                let l = direction;
                let n_dot_l = dot(n, *l);

                if n_dot_l > 0.0 {
                    i += intensity * n_dot_l / (length(n) * length(*l))
                }
            }
        }
    }

    i
}

fn canvas_to_viewport(
    (cx, cy): (f64, f64),
    (cw, ch): (f64, f64),
    (vw, vh): (f64, f64),
    d: f64,
) -> (f64, f64, f64) {
    let vx = cx * (vw / cw);
    let vy = cy * (vh / ch);
    let vz = d;

    (vx, vy, vz)
}
