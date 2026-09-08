import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";

const config: Config = {
  title: "Svod",
  tagline: "Rust-based ML compiler with UOp IR",
  favicon: "img/favicon.ico",

  // `v4` turns off every MDX v1 compatibility shim, `mdx1Compat.admonitions`
  // among them, so admonition titles must use the directive-label form
  // (`:::tip[Title]`); the legacy `:::tip Title` renders its title as body text.
  future: {
    v4: true,
  },

  url: "https://svod.vpermilp.online",
  baseUrl: "/",

  organizationName: "Patsakula Nikita",
  projectName: "svod",

  onBrokenLinks: "throw",

  // The docs are plain CommonMark (no imports/JSX) — parse `.md` as Markdown,
  // not MDX, so explicit heading IDs (`{#id}`, used by translations to keep
  // anchors stable) and stray `{`/`<` in prose don't trip MDX expression
  // parsing. `.mdx` (if added later) still gets full MDX.
  markdown: {
    format: "detect",
    mermaid: true,
  },

  i18n: {
    defaultLocale: "en",
    locales: ["en", "zh-Hans", "ru", "hi"],
    localeConfigs: {
      en: {
        label: "English",
      },
      "zh-Hans": {
        htmlLang: "zh-CN",
        label: "简体中文",
      },
      ru: {
        htmlLang: "ru",
        label: "Русский",
      },
      hi: {
        htmlLang: "hi",
        label: "हिन्दी",
      },
    },
  },

  plugins: [
    './plugins/readme-intro.mjs',
  ],

  themes: [
    "@docusaurus/theme-mermaid",
    [
      "@easyops-cn/docusaurus-search-local",
      {
        hashed: true,
        language: ["en", "zh", "ru", "hi"],
        indexBlog: false,
        highlightSearchTermsOnTargetPage: true,
      },
    ],
  ],

  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: "./sidebars.ts",
          editUrl: "https://github.com/npatsakula/svod/edit/main/website/",
        },
        blog: false,
        theme: {
          customCss: "./src/css/custom.css",
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: "img/docusaurus-social-card.jpg",
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: "Svod",
      logo: {
        alt: "Svod Logo",
        src: "img/logo.svg",
      },
      items: [
        {
          type: "docSidebar",
          sidebarId: "defaultSidebar",
          position: "left",
          label: "Docs",
        },
        {
          type: "localeDropdown",
          position: "right",
        },
        {
          href: "https://github.com/npatsakula/svod",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Docs",
          items: [
            {
              label: "Introduction",
              to: "/docs/introduction",
            },
          ],
        },
        {
          title: "Community",
          items: [
            {
              label: "GitHub",
              href: "https://github.com/npatsakula/svod",
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Svod. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ["rust", "cpp"],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
