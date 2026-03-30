import { homedir } from "os";
import { join } from "path";

export function getTargetPaths(scope) {
  const base =
    scope === "global"
      ? join(homedir(), ".claude")
      : join(process.cwd(), ".claude");

  return {
    commands: join(base, "commands", "turing"),
    agents: join(base, "commands", "turing", "agents"),
    config: join(base, "commands", "turing", "config"),
    claudeMd:
      scope === "global"
        ? join(homedir(), ".claude", "CLAUDE.md")
        : join(process.cwd(), "CLAUDE.md"),
    scope,
  };
}
