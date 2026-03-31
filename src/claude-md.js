import { readFile, writeFile } from "fs/promises";

const BEGIN = "<!-- BEGIN turing -->";
const END = "<!-- END turing -->";

const SECTION = `${BEGIN}
## turing

Autonomous ML research harness. The autoresearch loop as a formal protocol.

### Commands

| Command | Purpose |
|---------|---------|
| \`/turing\` | Router — detects ML intent and routes to sub-commands |
| \`/turing:init\` | Scaffold a new ML project with autoresearch harness |
| \`/turing:train [N]\` | Run autonomous experiment loop (optional max iterations) |
| \`/turing:status\` | Show experiment status, best model, convergence state |
| \`/turing:compare <a> <b>\` | Side-by-side experiment comparison |
| \`/turing:sweep\` | Generate and run hyperparameter sweep |
| \`/turing:validate\` | Check metric stability, auto-fix if noisy |
| \`/turing:try <hypothesis>\` | Inject a hypothesis into the experiment queue |
| \`/turing:brief\` | Generate research intelligence report |

### Agents

| Agent | Purpose |
|-------|---------|
| \`@ml-researcher\` | Autonomous training agent (Read/Write/Edit/Bash) |
| \`@ml-evaluator\` | Read-only analysis agent (Read/Bash only) |
${END}`;

export async function updateClaudeMd(claudeMdPath) {
  let content = "";
  try {
    content = await readFile(claudeMdPath, "utf-8");
  } catch {
    // File doesn't exist yet
  }

  const regex = new RegExp(
    `${BEGIN.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}[\\s\\S]*?${END.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`,
  );

  if (regex.test(content)) {
    content = content.replace(regex, SECTION);
  } else {
    content = content
      ? content.trimEnd() + "\n\n" + SECTION + "\n"
      : SECTION + "\n";
  }

  await writeFile(claudeMdPath, content, "utf-8");
}
