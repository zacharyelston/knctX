fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "kinect2")]
    {
        println!("cargo:rerun-if-changed=csrc/kinect2_shim.cpp");
        let mut build = cc::Build::new();
        build.cpp(true).std("c++14").file("csrc/kinect2_shim.cpp").warnings(false);
        match pkg_config::Config::new().probe("freenect2") {
            Ok(lib) => {
                for p in lib.include_paths { build.include(p); }
            }
            Err(_) => {
                // Fall back to the usual install locations.
                build.include("/usr/include").include("/usr/local/include");
                println!("cargo:rustc-link-search=native=/usr/local/lib");
                println!("cargo:rustc-link-lib=dylib=freenect2");
            }
        }
        build.compile("kinect2_shim");
    }
}
