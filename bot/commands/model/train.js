const { SlashCommandBuilder } = require('discord.js');
const axios = require('axios');
const fs = require('fs');


// Command data.
const data = new SlashCommandBuilder()
    .setName('train')
    .setDescription('Train the server model!')


// Command execution function.
async function execute(message) {
    let url = `${process.env.API_URL}/train/${message.guild.id}`;

    await axios.post(url).then(res => {
        if (res.status == 200) {
            message.reply(`Training started with guild ID ${message.guild.id} and user ID ${message.user.id} (${res.data.num_categories} categories).`);
        } else {
            message.reply("Error: " + res.data);
        }
    })
}

module.exports = {
    data: data,
    execute: execute
}

