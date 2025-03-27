const { SlashCommandBuilder } = require('discord.js');


// Command data.
const data = new SlashCommandBuilder()
    .setName('ping')
    .setDescription('Replies with Pong.');

// Command execution function.
async function execute(message) {
  await message.reply('Pong.');
  return;
}

module.exports = {
    data: data,
    execute: execute
}

