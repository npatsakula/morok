import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";

const config: Config = {
  title: "Morok",
  tagline: "Rust-based ML compiler with UOp IR",
  favicon: "img/favicon.ico",

  future: {
    v4: true,
  },

  url: "https://docs.morok.tech",
  baseUrl: "/",

  organizationName: "Patsakula Nikita",
  projectName: "morok",

  onBrokenLinks: "throw",

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

  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: "./sidebars.ts",
          editUrl: "https://github.com/npatsakula/morok/edit/main/website/",
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
      title: "Morok",
      logo: {
        alt: "Morok Logo",
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
          href: "https://github.com/npatsakula/morok",
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
              href: "https://github.com/npatsakula/morok",
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Morok. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ["rust"],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
