import { cva } from "class-variance-authority";

/**
 * SAFEOPS button styles. Applied to <button> and <Link> elements alike:
 *   <Link to="/dashboard" className={buttonVariants({ variant: "primary" })}>
 */
export const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 font-mono text-xs uppercase tracking-[0.16em] transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring active:translate-y-px disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        primary:
          "bg-primary text-primary-foreground shadow-[0_10px_26px_-12px_oklch(0.78_0.15_72/55%)] hover:bg-primary/90 hover:shadow-[0_14px_32px_-12px_oklch(0.78_0.15_72/65%)]",
        outline:
          "border border-input bg-secondary/25 text-foreground hover:border-primary/40 hover:bg-secondary hover:text-foreground",
        glass:
          "rounded-lg border border-glass-border bg-glass text-foreground backdrop-blur-md hover:border-foreground/30 hover:bg-glass-strong",
        ghost: "text-muted-foreground hover:bg-secondary hover:text-foreground",
      },

      size: {
        sm: "h-8 px-3 text-[10px]",
        default: "h-10 px-5",
        lg: "h-12 px-7 text-[13px]",
      },
    },
    defaultVariants: {
      variant: "primary",
      size: "default",
    },
  },
);
