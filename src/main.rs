// https://www.gabrielgambetta.com/computer-graphics-from-scratch/basic-ray-tracing.html

extern crate cgmath;
extern crate png;

use cgmath::*;

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const BACKGROUND_COLOR: Vector4<u8> = vec4(255, 255, 255, 255);

struct Scene {
    camera_pos: Vector3<f64>,
    camera_rotation: Matrix3<f64>,
    camera_d: f64,
    viewport_size: Vector2<f64>,
    spheres: Vec<Sphere>,
    lights: Vec<Light>,
}

struct Sphere {
    center: Vector3<f64>,
    radius: f64,
    color: Vector3<u8>,
    specular: f64,
    reflective: f64,
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
        position: Vector3<f64>,
    },
    Directional {
        intensity: f64,
        direction: Vector3<f64>,
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

    pub fn put_pixel_rgba(&mut self, position: Vector2<usize>, rgba: Vector4<u8>) {
        let idx = (position.y * self.width + position.x) * 4;
        self.data[idx + 0] = rgba.x;
        self.data[idx + 1] = rgba.y;
        self.data[idx + 2] = rgba.z;
        self.data[idx + 3] = rgba.w;
    }
}

pub fn main() {
    let mut texture = Texture::new(512, 512);

    let scene = Scene {
        camera_pos: vec3(3.0, 0.0, 1.0), // O
        camera_rotation: Matrix3::look_at(vec3(1.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0)),
        camera_d: 1.0,                 // d, Distance from camera to viewport
        viewport_size: vec2(1.0, 1.0), // vw, vh
        spheres: vec![
            Sphere {
                center: vec3(0.0, -1.0, 3.0),
                radius: 1.0,
                color: vec3(255, 0, 0), // Red
                specular: 500.0,        // Shiny
                reflective: 0.2,        // A bit reflective
            },
            Sphere {
                center: vec3(2.0, 0.0, 4.0),
                radius: 1.0,
                color: vec3(0, 0, 255), // Blue
                specular: 500.0,        // Shiny
                reflective: 0.3,        // A bit more reflective
            },
            Sphere {
                center: vec3(-2.0, 0.0, 4.0),
                radius: 1.0,
                color: vec3(0, 255, 0), // Green
                specular: 10.0,         // Somewhat shiny
                reflective: 0.4,        // Even more reflective
            },
            Sphere {
                center: vec3(0.0, -5001.0, 0.0),
                radius: 5000.0,
                color: vec3(255, 255, 0), // Yellow
                specular: 1000.0,         // Very shiny
                reflective: 0.5,          // Half reflective
            },
        ],
        lights: vec![
            Light::Ambient { intensity: 0.2 },
            Light::Point {
                intensity: 0.6,
                position: vec3(2.0, 1.0, 0.0),
            },
            Light::Directional {
                intensity: 0.2,
                direction: vec3(1.0, 4.0, 4.0),
            },
        ],
    };

    for cy in -(texture.height as i32 / 2)..(texture.height as i32 / 2) {
        for cx in -(texture.width as i32 / 2)..(texture.width as i32 / 2) {
            let d = scene.camera_rotation
                * canvas_to_viewport(
                    vec2(cx as f64, cy as f64),
                    vec2(texture.width as f64, texture.height as f64),
                    scene.viewport_size,
                    scene.camera_d,
                );
            let rgba = trace_ray(&scene, scene.camera_pos, d, 1.0, std::f64::INFINITY, 15);

            texture.put_pixel_rgba(
                vec2(
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

fn closest_intersection(
    scene: &Scene,
    o: Vector3<f64>,
    d: Vector3<f64>,
    t_min: f64,
    t_max: f64,
) -> (Option<&Sphere>, f64) {
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

    (closest_sphere, closest_t)
}

fn trace_ray(
    scene: &Scene,
    o: Vector3<f64>,
    d: Vector3<f64>,
    t_min: f64,
    t_max: f64,
    recursion_depth: u16,
) -> Vector4<u8> {
    let (closest_sphere, closest_t) = closest_intersection(scene, o, d, t_min, t_max);

    match closest_sphere {
        None => BACKGROUND_COLOR,
        Some(sphere) => {
            // Compute local color
            let p = o + d * closest_t; // Compute intersection
            let mut n = p - sphere.center; // Compute sphere normal at intersection
            n = n.normalize(); // Normalize

            let lighting = compute_lighting(scene, p, n, -d, sphere.specular);
            let rgb = sphere.color;

            let local_color = vec3(
                rgb.x as f64 * lighting,
                rgb.y as f64 * lighting,
                rgb.z as f64 * lighting,
            );

            // If we hit the recursion limit or the object is not reflective, we're done
            let reflective = sphere.reflective;

            if recursion_depth == 0 || reflective <= 0.0 {
                return vec4(
                    local_color.x as u8,
                    local_color.y as u8,
                    local_color.z as u8,
                    255,
                );
            }

            // Compute the reflected color
            let reflect_ray = reflect_ray(-d, n);
            let reflected_color = trace_ray(
                scene,
                p,
                reflect_ray,
                0.001,
                std::f64::INFINITY,
                recursion_depth - 1,
            );

            // local_color * (1 - r) + reflected_color * r

            vec4(
                (local_color.x * (1.0 - reflective) + (reflected_color.x as f64) * reflective)
                    as u8,
                (local_color.y * (1.0 - reflective) + (reflected_color.y as f64) * reflective)
                    as u8,
                (local_color.z * (1.0 - reflective) + (reflected_color.z as f64) * reflective)
                    as u8,
                255,
            )
        }
    }
}

fn reflect_ray(r: Vector3<f64>, n: Vector3<f64>) -> Vector3<f64> {
    n * 2.0 * dot(n, r) - r
}

fn intersect_ray_sphere(o: Vector3<f64>, d: Vector3<f64>, sphere: &Sphere) -> (f64, f64) {
    let c = sphere.center;
    let r = sphere.radius;
    let oc = o - c;

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

fn compute_lighting(
    scene: &Scene,
    p: Vector3<f64>,
    n: Vector3<f64>,
    v: Vector3<f64>,
    specular: f64,
) -> f64 {
    let mut i = 0.0;

    for light in &scene.lights {
        let (intensity, l, t_max) = match light {
            Light::Ambient { intensity } => {
                i += intensity;
                continue;
            }
            Light::Point {
                intensity,
                position,
            } => (intensity, *position - p, 1.0),
            Light::Directional {
                intensity,
                direction,
            } => (intensity, *direction, std::f64::INFINITY),
        };

        // Shadow check
        let (shadow_sphere, _shadow_t) = closest_intersection(scene, p, l, 0.001, t_max);
        if let Some(_) = shadow_sphere {
            continue;
        }

        // Diffuse
        let n_dot_l = dot(n, l);

        if n_dot_l > 0.0 {
            i += intensity * n_dot_l / (n.magnitude() * l.magnitude())
        }

        // Specular
        if specular != -1.0 {
            let r = reflect_ray(l, n);
            let r_dot_v = dot(r, v);

            if r_dot_v > 0.0 {
                i += intensity * (r_dot_v / (r.magnitude() * v.magnitude())).powf(specular);
            }
        }
    }

    if i < 0.0 {
        0.0
    } else if i > 1.0 {
        1.0
    } else {
        i
    }
}

fn canvas_to_viewport(
    point: Vector2<f64>,
    canvas_size: Vector2<f64>,
    viewport_size: Vector2<f64>,
    d: f64,
) -> Vector3<f64> {
    let vx = point.x * (viewport_size.x / canvas_size.x);
    let vy = point.y * (viewport_size.y / canvas_size.y);
    let vz = d;

    vec3(vx, vy, vz)
}
