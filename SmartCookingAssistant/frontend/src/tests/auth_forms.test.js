const fs = require("fs");
const path = require("path");

describe("Auth form UI", () => {
  test("login form renders expected controls", () => {
    const loginHtml = fs.readFileSync(path.join(__dirname, "../../login.html"), "utf-8");
    expect(loginHtml).toContain('id="email"');
    expect(loginHtml).toContain('id="password"');
    expect(loginHtml).toContain('id="login-btn"');
  });

  test("signup form has validation message path", () => {
    const signupHtml = fs.readFileSync(path.join(__dirname, "../../signup.html"), "utf-8");
    expect(signupHtml).toContain('id="name"');
    expect(signupHtml).toContain('id="email"');
    expect(signupHtml).toContain('id="password"');
    expect(signupHtml).toContain("Name, email, and password are required.");
  });
});
