import argparse

from sim2sim.visualization import SpherePushingContactForceVisualizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        required=True,
        type=str,
        help="Path to the experiment data folder.",
    )
    parser.add_argument(
        "--hydroelastic",
        action="store_true",
        help="Whether to plot hydroelastic or point contact forces.",
    )
    parser.add_argument(
        "--kIllustration",
        action="store_true",
        help="Whether to use kIllustration or kProximity for meshcat.",
    )
    parser.add_argument(
        "--manipuland",
        default="both",
        type=str,
        help="The manipuland to visualize. Options are 'outer', 'inner', 'both', and 'none'.",
    )
    parser.add_argument(
        "--newtons_per_meter",
        default=1e2,
        type=float,
        help="Sets the length scale of the force vectors.",
    )
    parser.add_argument(
        "--newton_meters_per_meter",
        default=1.0,
        type=float,
        help="Sets the length scale of the torque/ moment vectors.",
    )
    parser.add_argument(
        "--separation_distance",
        default=0.0,
        type=float,
        help="The distance in meters that the outer and inner manipuland should be separated from each other. "
        + "This only has an effect if `--manipuland` is 'both'.",
    )
    parser.add_argument(
        "--sphere_transparency",
        default=1.0,
        type=float,
        help="The alpha value of the sphere in range (0,1].",
    )
    parser.add_argument(
        "--save_html", action="store_true", help="Whether to save the meshcat HTML."
    )
    args = parser.parse_args()

    visualizer = SpherePushingContactForceVisualizer(
        data_path=args.data,
        manipuland=args.manipuland,
        separation_distance=args.separation_distance,
        save_html=args.save_html,
        newtons_per_meter=args.newtons_per_meter,
        newton_meters_per_meter=args.newton_meters_per_meter,
        hydroelastic=args.hydroelastic,
        kIllustration=args.kIllustration,
        sphere_transparency=args.sphere_transparency,
    )
    visualizer.setup()

    # NOTE: This starts an infinite loop
    visualizer.run()


if __name__ == "__main__":
    main()
