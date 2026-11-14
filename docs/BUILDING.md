# Building Denoise

You need macOS 14 or later on Apple Silicon, the Command Line Tools, Python 3, and Adobe's Premiere Pro and After Effects C++ SDKs (free from Adobe). You don't need Xcode.

Tell the build where the SDKs are, in your environment or in a `local.mk` file:

```make
PREMIERE_SDK ?= /path/to/Premiere Pro 26.0 C++ SDK
AE_SDK ?= /path/to/AdobeAfterEffectsSDK_26.5_MacOS
```

Quit Premiere, then:

```sh
make plugin          # builds build/Denoise.plugin
scripts/install.sh   # installs it
```

`make test` runs the tests.
