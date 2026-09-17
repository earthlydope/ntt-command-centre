/**
 * A fence around one chart.
 *
 * WHY. A D3 module runs inside an effect, and an exception thrown there walks
 * up to the nearest error boundary — which, until this file, was the root of
 * the app. One chart hitting a shape it could not draw blanked the whole growth
 * page: the KPIs, the brief, the action rail and the four charts that were
 * fine all disappeared behind React's empty root. That is the wrong blast
 * radius by several orders of magnitude, so every card in the evidence grid
 * now sits inside its own boundary and a failure is one calm card wide.
 *
 * WHAT IT DOES. On error it renders a card in the same chrome as the chart it
 * replaces — the title stays, so the reader still knows which question is
 * unanswered — with the message in a muted code line and a real "Try again"
 * that resets the boundary and lets the module have another go. The error is
 * logged once, with the chart named, so the console says which module fell
 * over rather than only that something did.
 *
 * `resetKey` is the spec object itself: a new payload (a filter change, a
 * refetch) hands the boundary a new object, and it clears the error on its own
 * so a chart that failed on one slice is retried on the next without a click.
 *
 * A class, because React exposes error boundaries only through
 * getDerivedStateFromError / componentDidCatch; there is no hook for this.
 */
import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  /** The chart's title, printed on the fallback so the gap is named. */
  title: string;
  /** When this identity changes the boundary forgets its error and remounts the child. */
  resetKey?: unknown;
  children: ReactNode;
}

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Once, here, with the chart named. React's own re-throw in development
    // will print the stack as well; this line is the one that says WHICH card.
    console.error(`Chart "${this.props.title}" could not be drawn:`, error, info.componentStack);
  }

  componentDidUpdate(prev: Props) {
    if (this.state.error !== null && prev.resetKey !== this.props.resetKey) {
      this.setState({ error: null });
    }
  }

  private reset = () => {
    this.setState({ error: null });
  };

  render() {
    const { error } = this.state;
    if (error === null) return this.props.children;

    const message = error.message && error.message.length > 0 ? error.message : String(error);
    return (
      <section className="card chart-fallback" role="alert" aria-label={`${this.props.title}: chart not drawn`}>
        <header className="card-head">
          <div className="card-head__text">
            <h3>{this.props.title}</h3>
            <p className="csub">This chart could not be drawn</p>
          </div>
        </header>
        <div className="card-empty chart-fallback__body">
          <p>
            The module for this chart stopped on an error. Everything else on the page is
            unaffected.
          </p>
          <code className="chart-fallback__err">{message}</code>
          <button type="button" className="pv-error__retry chart-fallback__retry" onClick={this.reset}>
            Try again
          </button>
        </div>
      </section>
    );
  }
}
