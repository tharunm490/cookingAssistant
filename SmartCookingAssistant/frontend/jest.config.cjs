module.exports = {
  testEnvironment: "jsdom",
  testMatch: ["<rootDir>/src/tests/**/*.test.js"],
  collectCoverageFrom: ["<rootDir>/app.js"],
  coverageDirectory: "coverage",
};
