const { SlashCommandBuilder } = require('discord.js');
const axios = require('axios');
const fs = require('fs');


// Command data.
const data = new SlashCommandBuilder()
    .setName('mark')
    .setDescription('Classify a message to a specific category')
    .addStringOption(option => 
        option.setName("category")
        .setDescription("The category to classify the message")
        .setRequired(true)
    )
    .addStringOption(option =>
        option.setName("message_id")
        .setDescription("The message ID to classify")
        .setRequired(false)
    )


// Command execution function.
async function execute(message) {
    const category = message.options.getString('category');
    const message_id = message.options.getString('message_id') ?? null;

    // get target message data
    let target_message = null


    if (!message_id) {
        //get last non-bot message
        target_message = await message.channel.messages.fetch({ limit: 10, before: message.id }).then(messages => {
            let non_bot_messages = messages.filter(m => !m.author.bot);
            if (non_bot_messages.size > 0) {
                return non_bot_messages.first();
            } else {
                message.reply("I couldn't find any non-bot messages in this channel. Please provide a valid message ID.");
                return null;
            }
        })

    } else {
        target_message = await message.channel.messages.fetch(message_id).catch(err => {
            message.reply("I couldn't find that message. Please provide a valid message ID.");
            return null;
        })
    }

    // send data to API
    let url = process.env.API_URL + '/message'
    let data = {
        message: target_message.content,
        category: category,
        user: target_message.member.id,
        guild: target_message.guild.id,
        timestamp: target_message.createdTimestamp.toString()
    };

    // axios post
    await axios.post(url, data).then(res => {
        if (res.status == 200) {
            message.reply('Message classified successfully!');
        } else {
            message.reply('Error classifying message: ' + res.statusText);
        }
    }).catch(err => {
        console.log(err);
        message.reply('Error classifying message: ' + err.message);
    });


    

  
}

module.exports = {
    data: data,
    execute: execute
}

