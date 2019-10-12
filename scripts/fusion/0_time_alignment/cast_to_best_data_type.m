function out_data = cast_to_best_data_type(str_cell_data)
    out_data = cell(size(str_cell_data));
    for col_index=1:size(str_cell_data,2)
        is_numeric = 0;
        is_boolean = 0;
        [dummy, is_numeric] = str2num(str_cell_data{1,col_index});
        if ~is_numeric
            is_boolean = strncmpi(str_cell_data{1,col_index}, 'f', 1) | strncmpi(str_cell_data{1,col_index}, 't', 1);
        end
        if is_numeric
            out_data(:,col_index) = num2cell(cellfun(@str2num, str_cell_data(:,col_index)));
        elseif is_boolean
            out_data(:,col_index) = num2cell(cellfun(@(x) lower(x(1))=='t'*1.0, str_cell_data(:,col_index)));
        else
            out_data(:,col_index) = str_cell_data{:,col_index};
        end
    end
    
    try
        out_data = cell2mat(out_data);
    catch
        out_data = cellfun(@(x) x+0, out_data);
    end
end