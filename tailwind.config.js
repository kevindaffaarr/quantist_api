/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./pages/**/*.{html,js}"],
  theme: {
    extend: {},
  },
  plugins: [require("daisyui")],
  daisyui: {
    themes: ['dark']
  }
}
