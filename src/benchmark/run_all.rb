#!/usr/bin/ruby

if(ENV["MAIN_DIR"])
    main_folder = ENV["MAIN_DIR"]
else
    main_folder = "../../"
end

folders = Dir["#{main_folder}*/"]

folders = folders.select { |item| item.include? "/build" }

folders.each do |item|
    puts "Benchmarking for build #{item}"
    `BIN_DIR=#{item} ./run.rb`
end
