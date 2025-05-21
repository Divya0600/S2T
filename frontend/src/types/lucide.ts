import { LucideProps } from 'lucide-react';
import { ForwardRefExoticComponent, RefAttributes, ComponentType } from 'react';

/**
 * Type definition for Lucide icons that's compatible with both direct imports
 * and JSX usage in the application
 */
export type LucideIconComponent = ForwardRefExoticComponent<Omit<LucideProps, "ref"> & RefAttributes<SVGSVGElement>>;

/**
 * Type for the icon prop in Card and StatusCard components
 */
export type LucideIcon = LucideIconComponent;
