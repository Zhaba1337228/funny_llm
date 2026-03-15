import type { Config } from 'tailwindcss'

const config: Config = {
  darkMode: ['class'],
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      boxShadow: {
        soft: '0 24px 60px rgba(8, 15, 31, 0.28)',
        glow: '0 0 0 1px rgba(255,255,255,0.08), 0 20px 60px rgba(21, 190, 166, 0.14)',
      },
      colors: {
        ink: {
          950: '#07111f',
          900: '#0a1528',
          800: '#11203a',
        },
      },
      fontFamily: {
        sans: ['"Manrope"', 'system-ui', 'sans-serif'],
        display: ['"Space Grotesk"', '"Manrope"', 'system-ui', 'sans-serif'],
      },
      backgroundImage: {
        mesh: 'radial-gradient(circle at top left, rgba(41, 255, 201, 0.18), transparent 32%), radial-gradient(circle at top right, rgba(83, 161, 255, 0.2), transparent 28%), radial-gradient(circle at bottom left, rgba(255, 176, 104, 0.16), transparent 26%)',
      },
      borderRadius: {
        xl2: '1.5rem',
      },
    },
  },
  plugins: [],
}

export default config
