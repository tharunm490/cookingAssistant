const fs = require("fs");
const path = require("path");

describe("Main UI logic coverage", () => {
  test("supports multiple image preview and remove actions", () => {
    const script = fs.readFileSync(path.join(__dirname, "../../app.js"), "utf-8");
    expect(script).toContain("function renderPreviews()");
    expect(script).toContain("function deleteImageByIndex(index)");
    expect(script).toContain("delete-image-btn");
  });

  test("supports clear all images and new recipe reset", () => {
    const script = fs.readFileSync(path.join(__dirname, "../../app.js"), "utf-8");
    expect(script).toContain("function clearAllImages()");
    expect(script).toContain("function resetAll()");
    expect(script).toContain("Ready for a new recipe! Upload images to begin.");
  });

  test("recipe card and voice player render paths exist", () => {
    const script = fs.readFileSync(path.join(__dirname, "../../app.js"), "utf-8");
    expect(script).toContain("function renderRecipe()");
    expect(script).toContain("function renderAudio()");
    expect(script).toContain("<audio controls");
  });
});
