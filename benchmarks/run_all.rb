#!/usr/bin/ruby

require 'open3'

if(ENV["MAIN_DIR"])
    main_folder = ENV["MAIN_DIR"]
else
    main_folder = "../../"
end

folders = Dir["#{main_folder}*/"]

folders = folders.select { |item| item.include? "/build" }

folders.each do |item|
    puts "Benchmarking for build #{item}"
    result_name = File.basename item
    result_name.sub! 'build_', ''
    result_name.sub! 'build', ''
    if result_name.length == 0 then
        result_name = "result"
    end
    system("BIN_DIR=#{item} ./run.rb #{result_name}.csv")
end
