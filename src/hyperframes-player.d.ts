/**
 * Type declaration for the <hyperframes-player> Web Component custom element.
 * Uses the React 19 / react-jsx JSX runtime's IntrinsicElements extension point.
 */
import type { DetailedHTMLProps, HTMLAttributes } from "react";

declare module "react/jsx-runtime" {
  namespace JSX {
    interface IntrinsicElements {
      "hyperframes-player": DetailedHTMLProps<
        HTMLAttributes<HTMLElement> & {
          src?: string;
          controls?: boolean | "";
        },
        HTMLElement
      >;
    }
  }
}
