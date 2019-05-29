<img src='https://g.gravizo.com/svg?
digraph G {
  libface_prn_network[label="libface prn network" shape=box]
  libface_prn_output[label="libface prn output" shape=box]
  deform_vic_face_to_match_prn_output[label="deform vic face to match prn output" shape=box]
  head_alignment[label="head alignment" shape=box]
  face_embed[label="face embedding" shape=box]
  seam_solving[label="seam solving" shape=box]
  head_alignment->face_embed
  face_embed -> seam_solving
  libface_prn_network->libface_prn_output
  libface_prn_output->deform_vic_face_to_match_prn_output
  deform_vic_face_to_match_prn_output->face_embed
}'/>
