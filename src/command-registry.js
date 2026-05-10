import { readFile } from 'fs/promises';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import YAML from 'yaml';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PLUGIN_ROOT = dirname(__dirname);
const REGISTRY_PATH = join(PLUGIN_ROOT, 'config', 'commands.yaml');

const COMMAND_NAME_PATTERN = /^[a-z][a-z0-9-]*$/;
const INVOCATION_MODES = new Set(['slash_only']);
const MODEL_INVOCATIONS = new Set(['disabled', 'enabled']);
const SCRIPT_LOCATIONS = new Set(['repo', 'scaffold']);

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function requireRecord(value, label) {
  if (!isRecord(value)) {
    throw new Error(`${label} must be a mapping`);
  }
  return value;
}

function requireNonEmptyString(value, label) {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`${label} must be a non-empty string`);
  }
  return value;
}

function requireNonEmptyStringList(value, label) {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`${label} must be a non-empty string list`);
  }

  for (const [index, item] of value.entries()) {
    requireNonEmptyString(item, `${label}[${index}]`);
  }

  return value;
}

function requireEnum(value, allowed, label) {
  requireNonEmptyString(value, label);
  if (!allowed.has(value)) {
    throw new Error(`${label} must be one of: ${Array.from(allowed).join(', ')}`);
  }
  return value;
}

function validateEquivalentScript(value, commandName) {
  const label = `commands.${commandName}.equivalent_script`;
  const script = requireRecord(value, label);
  const keys = Object.keys(script).sort();
  const expectedKeys = ['location', 'path'];
  if (keys.length !== expectedKeys.length || keys.some((key, index) => key !== expectedKeys[index])) {
    throw new Error(`${label} must contain exactly: ${expectedKeys.join(', ')}`);
  }

  requireNonEmptyString(script.path, `${label}.path`);
  requireEnum(
    script.location,
    SCRIPT_LOCATIONS,
    `${label}.location`,
  );
}

function validateCommand(commandName, value) {
  if (!COMMAND_NAME_PATTERN.test(commandName)) {
    throw new Error(`Invalid command name: ${commandName}`);
  }

  const command = requireRecord(value, `commands.${commandName}`);
  requireNonEmptyString(command.description, `commands.${commandName}.description`);
  requireNonEmptyString(command.lifecycle, `commands.${commandName}.lifecycle`);
  requireEnum(command.invocation_mode, INVOCATION_MODES, `commands.${commandName}.invocation_mode`);
  requireEnum(
    command.model_invocation,
    MODEL_INVOCATIONS,
    `commands.${commandName}.model_invocation`,
  );

  if (typeof command.mutates_project !== 'boolean') {
    throw new Error(`commands.${commandName}.mutates_project must be a boolean`);
  }

  requireNonEmptyStringList(command.tools, `commands.${commandName}.tools`);

  if ('argument_hint' in command) {
    requireNonEmptyString(command.argument_hint, `commands.${commandName}.argument_hint`);
  }

  if ('equivalent_script' in command) {
    validateEquivalentScript(command.equivalent_script, commandName);
  }
}

function validateRegistry(value) {
  const registry = requireRecord(value, 'Command registry root');
  const configFiles = requireNonEmptyStringList(registry.config_files, 'config_files');
  const commands = requireRecord(registry.commands, 'commands');

  for (const [commandName, command] of Object.entries(commands)) {
    validateCommand(commandName, command);
  }

  return {
    commands,
    commandNames: Object.keys(commands).sort(),
    configFiles: [...configFiles].sort(),
  };
}

export async function loadCommandRegistry(registryPath = REGISTRY_PATH) {
  let source;
  try {
    source = await readFile(registryPath, 'utf8');
  } catch (error) {
    throw new Error(`Failed to read command registry at ${registryPath}: ${error.message}`);
  }

  let parsed;
  try {
    parsed = YAML.parse(source);
  } catch (error) {
    throw new Error(`Failed to parse command registry at ${registryPath}: ${error.message}`);
  }

  try {
    return validateRegistry(parsed);
  } catch (error) {
    throw new Error(`Invalid command registry at ${registryPath}: ${error.message}`);
  }
}

export async function getCommandNames(registryPath) {
  const registry = await loadCommandRegistry(registryPath);
  return registry.commandNames;
}

// Installed public layout under .claude/commands/turing.
export async function getExpectedCommandPaths(registryPath) {
  const names = await getCommandNames(registryPath);
  return ['SKILL.md', ...names.map((name) => `${name}/SKILL.md`)];
}

// Editable repository source layout.
export async function getExpectedSkillSourcePaths(registryPath) {
  const names = await getCommandNames(registryPath);
  return [
    'skills/turing/SKILL.md',
    ...names.map((name) => `skills/turing/${name}/SKILL.md`),
    'skills/turing/rules/loop-protocol.md',
  ];
}

// Generated repository compatibility layout.
export async function getExpectedLegacyCommandCompatPaths(registryPath) {
  const names = await getCommandNames(registryPath);
  return [
    'commands/turing.md',
    ...names.map((name) => `commands/${name}.md`),
    'commands/rules/loop-protocol.md',
  ];
}

export async function getConfigFiles(registryPath) {
  const registry = await loadCommandRegistry(registryPath);
  return registry.configFiles;
}
