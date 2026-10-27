# Election Night Lab (public website)

This deployment branch is isolated from the default branch of yromano/cqr. The default research repository is not modified.

- index.html: standalone browser election dashboard with Monte Carlo national seat distributions and optional city-level ACI diagnostics.
- server.py: a same-origin /api/turnout relay. Requests Mako on each refresh. It returns 503 if data cannot be verified rather than inventing an observation.

The Mako article supplies turnout, **not party votes**. The forecast is conditional on a dated prior. National seat ranges are simulated model scenarios, not guaranteed conformal prediction intervals. City-level adaptive conformal scores require uploaded historical and completed-locality results.

To run locally: python server.py. The server respects PORT and is ready for Render.
