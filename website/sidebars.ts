import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  defaultSidebar: [
    {
      type: 'doc',
      id: 'introduction',
      label: 'Introduction',
    },
    {
      type: 'category',
      label: 'Getting Started',
      items: ['examples'],
    },
    {
      type: 'category',
      label: 'Architecture',
      items: [
        'architecture/pipeline',
        {
          type: 'category',
          label: 'Codegen Pipeline',
          items: [
            'architecture/codegen/overview',
            'architecture/codegen/rangeify',
            'architecture/codegen/expander',
            'architecture/codegen/devectorizer',
            'architecture/codegen/linearizer',
            'architecture/codegen/worked-example',
          ],
        },
        'architecture/ir-design',
        'architecture/optimizations',
        'architecture/op-bestiary',
      ],
    },
  ],
};

export default sidebars;
