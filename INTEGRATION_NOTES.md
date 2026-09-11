# Spectral sky and camera-motion integration

The branch `integration/spectral-envmap-camera-motion` combines the AVMC base
`fcda29d8`, upstream master `77690e8d`, and upstream `motion_blur` `d83ba29cd`.
The original `avmc_base` checkout remains at its original commit.

## Behavior and compatibility

- `spectral_envmap` reads linear, uniformly spaced spectral sky data and an
  optional supplied sun spectrum. It supports wavelengths beyond the visible
  band and a visible analytic sun disk. It does not generate an atmosphere.
- Named EXR channels survive component conversion. The wavelength list follows
  loaded Bitmap channel order, not an interpretation of channel names.
- Exposed map edits rebuild sampling distributions and periodic halo columns,
  including their gradient connections. Spatial resizing is supported; the
  wavelength grid is fixed per emitter.
- Sun irradiance uses the exact projected disk solid angle. Sky and sun have
  separately bounded spectral support; emitted-ray sampling covers their union.
  Set explicit bounds on a `uniform` sun spectrum for infrared use.
- `twosidedarea` is retained. Its emitted-ray sampler now covers both sides with
  the corresponding factor of two in the weight.
- `streammesh` retains SGX's typed pointer properties, copies the input buffers,
  and uses upstream's new mesh construction API. Transforms, UV flipping, and
  vertex temperature interpolation are covered by a native regression.
- Camera motion is tested over stationary thermal geometry. Moving emissive
  targets and target-to-sensor atmospheric transport remain separate work.

Upstream removes `Texture::mean()`, changes mesh buffers to shaped tensors, and
removes ray differentials. The motion branch changes internal transforms to
`AnimatedTransform`. The custom plugins were adapted, but external C++ clients
must be rebuilt against this version. Match Mitsuba's native architecture flags
(`-march=native` here), as the Release SGX build does: Dr.Jit type alignment is
ABI-sensitive.

## Reproducing the build

Load the requested environment in each build/test shell:

```sh
module load SupportPackages/sgx/release/0.0.4.0 sgx/1.0.0 gcc/14
```

This checkout was built with GCC 14.2.1, SupportPackages Python 3.13.5, external
JPEG/PNG/TIFF/OpenEXR, bundled Embree/Dr.Jit, and these variants:
`scalar_rgb`, `scalar_spectral`, `llvm_ad_rgb`, `llvm_ad_spectral`.

Session paths:

- Source: `/tmp/mitsuba3-spectral-motion-integration`
- Build: `/data1/mitsuba3/build-spectral-motion-gcc14`
- Isolated install: `/data1/mitsuba3/build-integration-install`
- Test environment: `/data1/mitsuba3/build-integration-tools/env-gcc14.sh`

The test environment prepends the build libraries to `LD_LIBRARY_PATH` before
the module-provided libraries. Otherwise the installed release's older Dr.Jit
can be selected and fail with an undefined symbol. It also selects the build's
Python packages and the test virtual environment.

## Validation

- Complete Release build of all four configured variants.
- 259 pytest cases passed: animated transforms, perspective camera, thermal
  shutter integration, visibility/null surfaces, instances, blackbody,
  `twosidedarea`, existing `envmap`, and the new spectral emitter.
- Native SGX streammesh regression passed before and after the motion merge.
  It can be repeated with `python src/shapes/tests/check_streammesh.py BUILD`.
- Supplied `lwir_test_spectral_envmap.py` and `lwir_test_sun_disk.py` passed on
  LLVM spectral. Temporary copies explicitly convert NumPy scalars to Python
  floats for `Vector3f`; the originals in Downloads were not modified.
- Isolated installation succeeded. A separate CMake project using
  `find_package(mitsuba)` and `Mitsuba::mitsuba` compiled and passed the native
  streammesh regression against that install.

CUDA/OptiX and Metal runtime behavior was not validated by this CPU/LLVM build.
The OptiX merge preserves the visibility-record update and motion-pipeline
switching paths, but needs backend-specific validation before GPU deployment.

## Test-data dependency

`resources/data` required its own merge to retain both master's visibility
reference images and the motion branch's scenes. Its local merge commit is
`c90904e`, on branch `integration/spectral-camera-motion` in that submodule.
That commit must be made available in an appropriate data remote when sharing
or pushing the parent integration branch. No remote branches or installed SGX
release files were changed.
