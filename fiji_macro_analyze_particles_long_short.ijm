// Fiji / ImageJ macro
// Long + Short variants for edge-based binary segmentation and particle analysis
//
// Usage:
// 1) Open an image (supports RGB or .tif stacks).
// 2) Run this macro.
// 3) When prompted, choose Long or Short.
//
// Notes:
// - The scale is set globally: distance=2208 px corresponds to 1 mm.
// - "Read and Write Excel" requires the corresponding Fiji plugin.

macro "Analyze Particles (Long or Short)" {
    setBatchMode(true);

    Dialog.create("Choose macro version");
    Dialog.addChoice("Version", newArray("Long", "Short"), "Long");
    Dialog.show();
    version = Dialog.getChoice();

    if (version == "Long") {
        runLong();
    } else {
        runShort();
    }

    setBatchMode(false);
}

function preprocessCommon() {
    fileName = getTitle();

    if (endsWith(fileName, ".tif")) {
        run("Stack to RGB");
        close(fileName);
    }

    run("Set Scale...", "distance=2208 known=1 pixel=1 unit=mm global");
    run("Subtract Background...", "rolling=50 light");
    run("Sharpen");
    run("Enhance Contrast...", "saturated=20.0");
    run("Find Edges");
    run("Gaussian Blur...", "sigma=2");
    setOption("BlackBackground", false);
    run("Make Binary");
}

function runLong() {
    preprocessCommon();

    run("Close-");
    run("Invert");
    run("Close-");

    run("Set Measurements...", "display redirect=None decimal=3");
    run("Analyze Particles...", "size=235-Infinity pixel display exclude clear");
    run("Read and Write Excel");
}

function runShort() {
    preprocessCommon();

    run("Close-");

    run("Set Measurements...", "display redirect=None decimal=3");
    run("Analyze Particles...", "size=235-Infinity pixel display exclude clear");
    run("Read and Write Excel");
}
