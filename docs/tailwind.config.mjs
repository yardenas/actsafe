/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Noto Sans"],
      },
      typography: ({ theme }) => ({
        paper: {
          css: {
            "--tw-prose-body": theme("colors.slate.800"),
            "--tw-prose-bullets": theme("colors.slate.800"),
            "--tw-prose-pre-bg": theme("colors.pink.900"),
            "--tw-prose-links": theme("colors.blue.500"),
            a: {
              color: theme("colors.blue.500"),
              textDecoration: "none",
              "&:hover": {
                textDecoration: "underline",
              },
            },
          },
        },
      }),
    },
  },
  plugins: [require("@tailwindcss/typography")],
};
