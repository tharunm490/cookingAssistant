describe("App DOM smoke", () => {
  beforeEach(() => {
    document.body.innerHTML = `
      <input id="file-input" type="file" />
      <div id="dropzone"></div>
      <div id="preview-grid"></div>
      <button id="clear-images-button"></button>
      <button id="detect-button"></button>
      <button id="generate-button"></button>
      <button id="voice-button"></button>
      <button id="new-recipe-button"></button>
      <div id="ingredient-list"></div>
      <div id="recipe-output"></div>
      <div id="nutrition-output"></div>
      <div id="audio-output"></div>
      <div id="health-optimization-output"></div>
      <div id="status"></div>
      <select id="meal-type"><option value="lunch">lunch</option></select>
      <select id="diet-type"><option value="balanced">balanced</option></select>
      <select id="spice-level"><option value="medium">medium</option></select>
      <div id="health-goals-container"></div>
    `;
    jest.resetModules();
  });

  test("initial render and new recipe action keep UI stable", () => {
    expect(() => require("../../app.js")).not.toThrow();

    const status = document.querySelector("#status");
    const newRecipeBtn = document.querySelector("#new-recipe-button");
    newRecipeBtn.click();

    expect(status.textContent).toContain("Ready for a new recipe");
    expect(document.querySelector("#recipe-output").textContent).toContain("Generated recipe");
    expect(document.querySelector("#audio-output").textContent).toContain("Create voice narration");
  });
});
